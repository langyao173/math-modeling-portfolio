"""Frozen reference-benchmark charges used by the selection objective."""

from __future__ import annotations

import hashlib
import json

REFERENCE_RUNTIME_CHARGES: dict[str, float] = {
    "00c4d0f9338d41830c763493293345d99e4f434be929fd1cc01f9e53dcde0e9a": 0.4,
    "059857f4abac74399ee86a8cc79103901cc649e6ec16913896c714c15ca6ce04": 0.3,
    "09babbe8538d8d96ecad47c2924e80fa2c82c2246f6819b5713d6e1224230821": 0.5,
    "376638f41178098a635052d86f8479d04274796456b195c1340fc44550bd0372": 0.3,
    "389b1d9b313146ad8823b554fbe57545fd654811d9ec4f625d9509104140c0e7": 0.4,
    "3ea78d41cdd292d94ab3ed4f002da1797f8d46c1b778c182ae10779c22593bca": 0.4,
    "416308bf3ee86b945c7ce3ac73018b924d63cdc4efab48f29bb315964d4f5422": 0.4,
    "440744b8728fe7f7fd0bae2ae10808d83219bc61ffe9af4c90c97d34c2312601": 0.3,
    "450826ed151adea73ef5ccefa841d830a8116e81eb6fdd01b53244cb02e20a62": 0.3,
    "45308640ee89efc5a391df012486cf026617f0afef787221a2baee7bbcfa74f2": 0.3,
    "548efa9409fcb5fbd8dffe3a36fce9e5cff270a73a5d84134bf6f70bff266136": 0.4,
    "550ab47eed0321e2ca4b7da35286b2f838d7f486d2dea6c1cebf9540211916d1": 0.4,
    "57ff5146fa8fffc440d3cb693ec18afec26ec5f4b492a6f3ec34f4b909c93b91": 0.4,
    "60fdc853cfcd3f68cb712eb5229b9a8b07a0b5d46381a412934ec34f4df4e63b": 0.3,
    "696dc45571682760d205f6fd61afebc6651515013595358c2a17bd359aae78c5": 0.4,
    "7cd8b77e701e83656df50a442da857f646efceaac9f89c0eeeb6f8bdb99c8860": 0.4,
    "8887198099dffcc2a49aa5a21ff3451dada5e465763eda25d0963f778e06f2a7": 0.3,
    "8c03b01b735756b2057da8405182ddf9e6b079936ecbec525e0d17a8104fc988": 0.4,
    "8fbb256098ce3eda0a538ebade1838a13d2e55bbd4ac3760d5a22ad3059f8d5d": 0.4,
    "99692315cbd2d7dc4a2549ad2a096ba813ae161f1a36b92b7927113f8b63390c": 0.3,
    "9d5a114d2bbcb87fb5a18521e762b2955f7bda663f6042a5d0592925d104679d": 0.4,
    "a4cc06c180a18924c5955948b68f12a6f90762763eeb6befb8ac360eb4da014b": 0.4,
    "a9367cd694e963f79760a11950b7fc907429cde4935e1145b379cd427b0ed3fa": 0.3,
    "b297986df759df6ed6d11ad364e1484b60fa6897a01f38dd0551d690cbece4cd": 0.3,
    "b77a04c04cc21514693bf016509111bff66f698f2973eb9e556299adc67fd3a9": 0.4,
    "c984d039da86061398a2cbc03bbca7781c54f1574adce7cd830c1ec2e2c8b755": 0.4,
    "cf98d4b0a562850b106e6e58ad419c66aa99a94cf15fd47fccfe4e6d674b900a": 0.4,
    "d1887d606ba15b99b93252a76b38f879e87dbdb418a653d94cb2f08389bfd68a": 0.3,
    "db4b9a90dbd04e2984619bdcff4a64d7503d91f3123154bac4877fdda9cff6e4": 0.4,
    "dfa10a26f146d03db944d61dcdc892e956ba449130beb4ea6a194486123a073e": 0.3,
    "e1d8bd7322e7b775147a2d86b2e235cc984ec9e6ffaa4adc4925eb8eb44db67f": 0.3,
    "e3bdbbd1eb1db09b57cb78577a48e948c9c54524b624e473bd03d7fc20d54211": 0.4,
    "f0d7c717e297d7f24be6ee2a62a7fb8f6b9f4428acbcdda7311cc449aef2d0a1": 0.4,
    "f12f0fc20700df70743ef3b4f27089c45f45989a93fa2bd19c1886c98d609a81": 0.4,
    "f24726bdc10568e8091d64562d56faf3750df9ad9260929ab8d2642bc6345a7e": 0.4,
    "f3e16210340179977f7f82486dc8cfc3dc0e3be410e9095cddbbe96d99b9249f": 0.4,
}

REFERENCE_RUNTIME_PROTOCOL = {
    "benchmark_data": "canonical pre_match rows through 2010 (772 rows)",
    "environment": {
        "platform": "Windows-11-10.0.26200-SP0",
        "processor": "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel",
        "python": "3.12.11",
    },
    "estimator": "median of 3 complete final-development searches",
    "quantization": "ceil to 0.1 reference seconds",
}


def _reference_hash() -> str:
    payload = {
        "charges": REFERENCE_RUNTIME_CHARGES,
        "protocol": REFERENCE_RUNTIME_PROTOCOL,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


REFERENCE_RUNTIME_HASH = _reference_hash()


def reference_runtime_seconds(candidate_id: str) -> float:
    """Return an immutable reference charge, never live execution time."""
    try:
        return REFERENCE_RUNTIME_CHARGES[candidate_id]
    except KeyError as error:
        raise ValueError("Candidate is absent from the frozen runtime benchmark") from error


__all__ = [
    "REFERENCE_RUNTIME_CHARGES",
    "REFERENCE_RUNTIME_HASH",
    "REFERENCE_RUNTIME_PROTOCOL",
    "reference_runtime_seconds",
]
