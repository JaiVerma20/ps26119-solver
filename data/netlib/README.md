# Netlib LP benchmark files

Place uncompressed `.mps` files here (e.g. `afiro.mps`).

Where to get them:

* The official Netlib LP set (https://www.netlib.org/lp/data/) is stored in a
  compressed "emps" format. Convert with Netlib's `emps` program before use.
* Many open-source solver repositories ship plain-MPS copies of the small
  instances (for example `check/instances/afiro.mps` in the HiGHS repository).

Published optimal objective values are listed at
https://www.netlib.org/lp/data/readme. First target:

| instance | rows | cols | optimal objective |
|---|---|---|---|
| afiro | 27 | 32 | -4.6475314286E+02 |

```bash
./build/gpuopt data/netlib/afiro.mps --expect -464.7531428571
```
