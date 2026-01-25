# Security Policy

## Reporting a Vulnerability

Please **DO NOT** open a public issue for security vulnerabilities.

If you have discovered a security issue in Loclean, please report it by emailing **nxan2911@gmail.com** (or link to your profile).

We will acknowledge your report within 48 hours.

## Known Vulnerabilities

### Protobuf JSON Recursion Depth Bypass (Unpatched)

**Status**: Unpatched vulnerability in transitive dependency  
**Affected Versions**: protobuf <= 6.33.4  
**Patched Version**: None (as of January 2025)  
**Severity**: Medium (DoS via RecursionError)

#### Description

A denial-of-service (DoS) vulnerability exists in `google.protobuf.json_format.ParseDict()` in Python, where the `max_recursion_depth` limit can be bypassed when parsing nested `google.protobuf.Any` messages. Due to missing recursion depth accounting inside the internal Any-handling logic, an attacker can supply deeply nested Any structures that bypass the intended recursion limit, eventually exhausting Python's recursion stack and causing a `RecursionError`.

#### Impact on Loclean

- **Direct Impact**: **None** - Loclean does not use `google.protobuf.json_format.ParseDict()` or `google.protobuf.Any` messages in its core codebase.
- **Indirect Impact**: **Low** - `protobuf` is only present as a transitive dependency of the optional `cloud` extra (via `google-generativeai`).
- **Affected Users**: Only users who install `loclean[cloud]` or `loclean[all]` are affected.

#### Mitigation

1. **Use C++ Implementation (Recommended)**: The vulnerability only affects the **Pure-Python** backend. The **C++ implementation** (default for PyPI wheels) is **not affected**. Ensure you're using the C++ implementation by setting:
   ```bash
   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
   ```

2. **Avoid Cloud Extra**: If you don't need cloud API support, avoid installing the `cloud` extra:
   ```bash
   # Instead of: pip install loclean[cloud]
   pip install loclean  # Core library only
   ```

3. **Input Validation**: If you're using `google-generativeai` directly, avoid parsing untrusted JSON/protobuf data from external sources.

#### Monitoring

We are monitoring upstream protobuf releases for a patch. Once a patched version is available, we will update dependencies accordingly.

**Last Updated**: January 24, 2025