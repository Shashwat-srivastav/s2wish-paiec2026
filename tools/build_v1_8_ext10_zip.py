"""Build submissions/dist/v1_8_ext10.zip: shipped v1_3 with E5's X1-ext10 offset prior (EXP-0057).

The only code change is the offset prior: a mixture of grid-normalised t3 components read from the new
`prior_mix` parameter, [[weight, centre, scale], ...]. With prior_mix absent the model is v1_3 unchanged.
Components follow src/paiec/models/x1_prior.py ext(CORE_H, 0.10): 0.90 x t3(G/2, learned), 0.05 x t3(+3, 1),
0.05 x t3(-3, 1). Pre-registration: docs/preregistrations/EXP-0057_v1_8_ext10_preregistration.md.
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submissions" / "dist" / "v1_3_named_facets_noguard.zip"
SRC_SHA = "a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"
OUT = ROOT / "submissions" / "dist" / "v1_8_ext10.zip"
FIXED_DATE = (1980, 1, 1, 0, 0, 0)

OLD_PRIOR = '''_Z = (MU - float(P["prior_center"])) / float(P["prior_scale"])
LOG_PRIOR = -2.0 * np.log1p(_Z * _Z / 3.0)'''

NEW_PRIOR = '''def _t3_log_norm(center, scale):
    z = (MU - float(center)) / float(scale)
    lp = -2.0 * np.log1p(z * z / 3.0)
    return lp - np.log(np.exp(lp - lp.max()).sum()) - lp.max()


if P.get("prior_mix"):
    # Mixture offset prior: grid-normalised t3 components, weights renormalised (EXP-0057, E5 X1-ext10).
    _wsum = sum(float(w) for w, _, _ in P["prior_mix"])
    LOG_PRIOR = None
    for _w, _c, _s in P["prior_mix"]:
        _lp = np.log(float(_w) / _wsum) + _t3_log_norm(_c, _s)
        LOG_PRIOR = _lp if LOG_PRIOR is None else np.logaddexp(LOG_PRIOR, _lp)
else:
    _Z = (MU - float(P["prior_center"])) / float(P["prior_scale"])
    LOG_PRIOR = -2.0 * np.log1p(_Z * _Z / 3.0)'''


def main():
    data = SRC.read_bytes()
    if hashlib.sha256(data).hexdigest() != SRC_SHA:
        raise SystemExit(f"{SRC.name} is not the shipped v1_3 archive")
    with zipfile.ZipFile(SRC) as z:
        files = {n: z.read(n) for n in z.namelist()}

    model = files["model.py"].decode("utf-8")
    if model.count(OLD_PRIOR) != 1:
        raise SystemExit("prior block not found exactly once in model.py")
    model = model.replace(OLD_PRIOR, NEW_PRIOR)
    old_default = '"guard": 1, "guard_rho": 0.5}'
    if model.count(old_default) != 1:
        raise SystemExit("_DEFAULT terminator not found exactly once")
    model = model.replace(old_default, '"guard": 1, "guard_rho": 0.5, "prior_mix": None}')
    files["model.py"] = model.encode("utf-8")

    params = json.loads(files["v1_2_params.json"])
    c, s = params["prior_center"], params["prior_scale"]
    params["prior_mix"] = [[0.90, c, s], [0.05, 3.0, 1.0], [0.05, -3.0, 1.0]]
    params["_source"] = ("v1_3_named_facets_noguard.zip (sha a4b0e841) + E5 X1-ext10 offset prior; "
                         "facets='named' behaves as all facets (truthy flag); guard=0")
    files["v1_2_params.json"] = (json.dumps(params, indent=1) + "\n").encode("utf-8")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])
    print(f"{OUT.relative_to(ROOT)}  sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
