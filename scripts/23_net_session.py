#!/usr/bin/env python3
"""Networked EncMonitor prototype: the in-process objects are replaced with serialized context,
key and ciphertext frames exchanged over TCP.

Three kinds of separate OS processes talk over TCP, using the framing of encmonitor.net.protocol
(length-prefixed JSON header + binary blob) and the serialization of encmonitor.he.serde:

  server     monitoring server: holds the context and evaluation keys (never the secret key), keeps one
             window state machine per window  CREATED -> COLLECTING -> READY -> COMPUTING -> COMPLETE | FAILED,
             evaluates the circuit (evaluate_dual_statistics + OpenFHEBackend) and returns
             RESULT ciphertexts. Duplicate uploads, unknown profile, wrong dimension, unknown window are hard
             errors; a timeout fails the window and never renormalizes weights over the sites that arrived.
  authority  key/decryption authority: generates context + key pair for the profile, sends INIT_CONTEXT,
             INIT_PUBLIC_KEY, INIT_EVAL_MULT, INIT_EVAL_ROT to the server, registers windows, requests COMPUTE,
             decrypts the RESULT ciphertexts. The secret key never leaves this process.
  sites      S site processes (one OS process per site): obtain the public context and public key from the
             server (GET_PUBLIC -> INIT_CONTEXT + INIT_PUBLIC_KEY), encrypt their own residual row locally and
             send one UPLOAD_RESIDUAL ciphertext per window.
  run        orchestrator: starts server, authority and sites as subprocesses on one host (loopback) or
             connects to a server on another host, then writes the session summary.

GET_PUBLIC and ACK are additions to the message schema (public material is distributed by the server;
uploads are acknowledged so a site knows its ciphertext was accepted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.net.protocol import recv_packet, send_packet


# ----------------------------------------------------------------------------------------------- helpers
def serde_public_key(pk) -> bytes:
    import openfhe as fhe
    from encmonitor.he import serde
    return serde._serialize(lambda p, t: fhe.SerializeToFile(p, pk, t), ".pk")


def deserde_public_key(blob: bytes):
    import openfhe as fhe
    from encmonitor.he import serde
    last = None
    for _, token in serde.token_matrix(True):
        path = serde._temp_blob(blob, ".pk")
        try:
            obj = serde._unwrap(fhe.DeserializePublicKey(path, token))
            if obj is not None:
                return obj
        except Exception as exc:
            last = exc
        finally:
            Path(path).unlink(missing_ok=True)
    raise RuntimeError(f"Public key deserialization failed: {last}")


def connect(host, port, retries=100):
    for _ in range(retries):
        try:
            return socket.create_connection((host, port), timeout=600)
        except OSError:
            time.sleep(0.1)
    raise ConnectionError(f"cannot connect to {host}:{port}")


class FrameLog:
    def __init__(self, path, role):
        self.path, self.role, self.lock = Path(path), role, threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, direction, header, stats, **extra):
        rec = {"role": self.role, "direction": direction, "type": header.get("type"),
               "window_id": header.get("window_id"), "site_id": header.get("site_id"),
               "header_bytes": stats.header_bytes, "blob_bytes": stats.blob_bytes, "wire_bytes": stats.wire_bytes,
               "t": time.time(), **extra}
        with self.lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------------------------- server
class Window:
    def __init__(self, run_id, window_id, expected, weights):
        self.run_id, self.window_id, self.expected = run_id, window_id, expected
        self.weights = np.asarray(weights, dtype=np.float64)
        self.state, self.cts, self.m = "COLLECTING", {}, None
        self.cond = threading.Condition()


def run_server(args):
    from encmonitor.he import serde
    from encmonitor.he.backend import OpenFHEBackend
    from encmonitor.he.circuit import evaluate_dual_statistics

    log = FrameLog(Path(args.out_dir) / "frames_server.jsonl", "server")
    st = {"profile_id": None, "cc": None, "context_blob": None, "pk_blob": None, "batch_size": None,
          "m": None, "windows": {}, "lock": threading.Lock(), "he_lock": threading.Lock(), "stop": False}

    def error(sock, hdr, code, detail):
        stats = send_packet(sock, {"type": "ERROR", "run_id": hdr.get("run_id"), "window_id": hdr.get("window_id"),
                                   "site_id": hdr.get("site_id"), "code": code, "detail": detail})
        log.add("send", {"type": "ERROR", "window_id": hdr.get("window_id")}, stats, code=code)

    def handle(sock):
        try:
            while True:
                hdr, blob, stats = recv_packet(sock)
                t = hdr.get("type")
                log.add("recv", hdr, stats)
                if t == "INIT_CONTEXT":
                    with st["he_lock"]:
                        st["cc"] = serde.deserialize_context(blob)
                    st.update(profile_id=hdr["profile_id"], context_blob=blob, batch_size=hdr["batch_size"], m=hdr["m"])
                elif t == "INIT_PUBLIC_KEY":
                    st["pk_blob"] = blob
                elif t == "INIT_EVAL_MULT":
                    with st["he_lock"]:
                        serde.deserialize_eval_mult(st["cc"], blob)
                elif t == "INIT_EVAL_ROT":
                    with st["he_lock"]:
                        serde.deserialize_eval_rot(st["cc"], blob)
                    send_packet(sock, {"type": "ACK", "of": "INIT"})
                elif t == "GET_PUBLIC":
                    for typ, b in (("INIT_CONTEXT", st["context_blob"]), ("INIT_PUBLIC_KEY", st["pk_blob"])):
                        s2 = send_packet(sock, {"type": typ, "profile_id": st["profile_id"], "batch_size": st["batch_size"], "m": st["m"]}, b)
                        log.add("send", {"type": typ}, s2)
                elif t == "REGISTER_WINDOW":
                    with st["lock"]:
                        st["windows"][hdr["window_id"]] = Window(hdr["run_id"], hdr["window_id"], hdr["expected_sites"], hdr["weights"])
                    send_packet(sock, {"type": "ACK", "of": "REGISTER_WINDOW", "window_id": hdr["window_id"]})
                elif t == "UPLOAD_RESIDUAL":
                    w = st["windows"].get(hdr.get("window_id"))
                    if hdr.get("profile_id") != st["profile_id"]:
                        error(sock, hdr, "PROFILE_MISMATCH", "ciphertext profile does not match the session context")
                    elif w is None:
                        error(sock, hdr, "UNKNOWN_WINDOW", "window not registered")
                    elif hdr.get("m") != st["m"]:
                        error(sock, hdr, "DIMENSION_MISMATCH", f"expected m={st['m']}")
                    elif not (0 <= int(hdr.get("site_id", -1)) < w.expected):
                        error(sock, hdr, "UNKNOWN_SITE", "site id outside the registered site set")
                    else:
                        with w.cond:
                            if w.state != "COLLECTING":
                                error(sock, hdr, "LATE_CIPHERTEXT", f"window state is {w.state}")
                            elif hdr["site_id"] in w.cts:
                                error(sock, hdr, "DUPLICATE_UPLOAD", "site already uploaded for this window")
                            else:
                                with st["he_lock"]:
                                    w.cts[hdr["site_id"]] = serde.deserialize_ciphertext(blob)
                                if len(w.cts) == w.expected:
                                    w.state = "READY"
                                    w.cond.notify_all()
                                send_packet(sock, {"type": "ACK", "of": "UPLOAD_RESIDUAL", "window_id": hdr["window_id"], "site_id": hdr["site_id"]})
                elif t == "COMPUTE":
                    w = st["windows"].get(hdr.get("window_id"))
                    if w is None:
                        error(sock, hdr, "UNKNOWN_WINDOW", "window not registered")
                        continue
                    with w.cond:
                        w.cond.wait_for(lambda: w.state == "READY", timeout=float(hdr.get("timeout_sec", args.timeout)))
                        if w.state != "READY":
                            w.state = "FAILED"
                            error(sock, hdr, "TIMEOUT", f"{len(w.cts)}/{w.expected} sites arrived; window FAILED (no renormalization)")
                            continue
                        w.state = "COMPUTING"
                    t0 = time.perf_counter()
                    with st["he_lock"]:
                        cts = [w.cts[s] for s in range(w.expected)]
                        res = evaluate_dual_statistics(OpenFHEBackend(st["cc"]), cts, w.weights, width=st["m"],
                                                       disclose_contributions=True, disclose_directional=False)
                    t_compute = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    outs = [("d_mix", None, res.d_mix), ("d_site", None, res.d_site), ("heterogeneity", None, res.heterogeneity)]
                    outs += [("contribution", s, c) for s, c in enumerate(res.contributions)]
                    with st["he_lock"]:
                        blobs = [(n, s, serde.serialize_ciphertext(c)) for n, s, c in outs]
                    t_ser = time.perf_counter() - t0
                    for n, s, b in blobs:
                        h = {"type": "RESULT", "run_id": w.run_id, "window_id": w.window_id, "score_name": n, "site_id": s,
                             "server_compute_sec": t_compute, "server_serialize_sec": t_ser, "n_results": len(blobs)}
                        log.add("send", h, send_packet(sock, h, b))
                    w.state = "COMPLETE"
                    w.cts.clear()
                elif t == "SHUTDOWN":
                    st["stop"] = True
                    send_packet(sock, {"type": "ACK", "of": "SHUTDOWN"})
                    return
                else:
                    error(sock, hdr, "UNKNOWN_MESSAGE", str(t))
        except (ConnectionError, OSError):
            return

    srv = socket.create_server((args.host, args.port), reuse_port=False)
    srv.settimeout(0.5)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "server_ready").write_text(str(os.getpid()))
    threads = []
    while not st["stop"]:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        th = threading.Thread(target=handle, args=(conn,), daemon=True)
        th.start()
        threads.append(th)
    srv.close()
    return 0


# ----------------------------------------------------------------------------------------------- authority
def load_windows(args):
    if args.replay:
        z = np.load(args.replay)
        R = np.asarray(z["residuals"], dtype=np.float64)[: args.windows]
        taus = z["tau_0.95"][: args.windows] if "tau_0.95" in z.files else None
        return R, np.asarray(z["weights"], dtype=np.float64), taus
    from encmonitor.synthetic import make_case
    geos = ["single", "coherent", "opposing", "mixed"]
    cases = [make_case(args.sites, args.m, geos[i % 4], 0.5, 2026 + i) for i in range(args.windows)]
    return np.stack([c.residuals for c in cases]), cases[0].weights, None


def run_authority(args):
    from encmonitor.he import serde
    from encmonitor.he.context import build_context, decrypt_scalar
    from encmonitor.statistics import compute_dual_statistics

    out = Path(args.out_dir)
    log = FrameLog(out / "frames_authority.jsonl", "authority")
    R, weights, taus = load_windows(args)
    N, S, m = R.shape
    t0 = time.perf_counter()
    bundle = build_context(m)
    t_keygen = time.perf_counter() - t0
    t0 = time.perf_counter()
    material = [("INIT_CONTEXT", serde.serialize_context(bundle.cc)), ("INIT_PUBLIC_KEY", serde_public_key(bundle.key_pair.publicKey)),
                ("INIT_EVAL_MULT", serde.serialize_eval_mult(bundle.cc)), ("INIT_EVAL_ROT", serde.serialize_eval_rot(bundle.cc))]
    t_serialize_keys = time.perf_counter() - t0
    profile_id = hashlib.sha256(material[0][1] + material[1][1]).hexdigest()[:16]
    sock = connect(args.host, args.port)
    t0 = time.perf_counter()
    for typ, blob in material:
        h = {"type": typ, "profile_id": profile_id, "run_id": args.run_id, "batch_size": bundle.batch_size,
             "ring_dim": bundle.ring_dimension, "m": m, "rotations": list(bundle.rotations)}
        log.add("send", h, send_packet(sock, h, blob))
    recv_packet(sock)  # ACK after INIT_EVAL_ROT: server has context + evaluation keys
    t_distribute = time.perf_counter() - t0
    for w in range(N):
        send_packet(sock, {"type": "REGISTER_WINDOW", "run_id": args.run_id, "window_id": w, "expected_sites": S,
                           "weights": weights.tolist(), "weights_hash": hashlib.sha256(weights.tobytes()).hexdigest()[:16]})
        recv_packet(sock)
    (out / "authority_ready").write_text(json.dumps({"profile_id": profile_id, "windows": N, "sites": S, "m": m}))

    g4 = {}
    if args.ckks_jsonl and Path(args.ckks_jsonl).exists():
        for line in open(args.ckks_jsonl):
            r = json.loads(line)
            g4[r["i"]] = r
    rows = []
    for w in range(N):
        t_req = time.perf_counter()
        send_packet(sock, {"type": "COMPUTE", "run_id": args.run_id, "window_id": w, "disclosure_level": "L3",
                           "timeout_sec": args.timeout})
        got, meta = {}, {}
        while len(got) < 3 + S:
            h, b, fs = recv_packet(sock)
            log.add("recv", h, fs)
            if h["type"] == "ERROR":
                raise RuntimeError(f"server error for window {w}: {h}")
            got[(h["score_name"], h["site_id"])] = b
            meta = h
        t_round = time.perf_counter() - t_req
        t0 = time.perf_counter()
        dec = {k: decrypt_scalar(bundle, serde.deserialize_ciphertext(v)) for k, v in got.items()}
        t_dec = time.perf_counter() - t0
        net = {"d_mix": dec[("d_mix", None)], "d_site": dec[("d_site", None)], "heterogeneity": dec[("heterogeneity", None)]}
        a_net = np.asarray([dec[("contribution", s)] for s in range(S)])
        ref = compute_dual_statistics(R[w], weights)
        err_plain = max([abs(net["d_mix"] - ref.d_mix), abs(net["d_site"] - ref.d_site), abs(net["heterogeneity"] - ref.heterogeneity)]
                        + list(np.abs(a_net - ref.contributions)))
        row = {"window": w, "net": net, "a_net": a_net.tolist(), "max_abs_err_vs_plaintext": float(err_plain),
               "top1_same_as_plaintext": int(np.argmax(a_net)) == int(np.argmax(ref.contributions)),
               "timing_sec": {"compute_round_trip": t_round, "server_compute": meta["server_compute_sec"],
                              "server_serialize": meta["server_serialize_sec"], "authority_decrypt": t_dec}}
        if w in g4:
            he = g4[w]["he"]
            row["max_abs_diff_vs_inprocess_ckks"] = float(max([abs(net[k] - he[k]) for k in net]
                                                               + list(np.abs(a_net - np.asarray(g4[w]["a_he"])))))
        if taus is not None:
            tau = taus[w]
            dn = [net["d_mix"] > tau[0], net["d_site"] > tau[1], net["heterogeneity"] > tau[2]]
            dp = [ref.d_mix > tau[0], ref.d_site > tau[1], ref.heterogeneity > tau[2]]
            row["decisions_agree_with_plaintext"] = bool(dn == dp)
        rows.append(row)
    with open(out / "authority_results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    setup = {"profile_id": profile_id, "ring_dimension": bundle.ring_dimension, "batch_size": bundle.batch_size,
             "keygen_sec": t_keygen, "serialize_keys_sec": t_serialize_keys, "distribute_keys_sec": t_distribute,
             "bytes": {typ: len(b) for typ, b in material}}
    (out / "authority_setup.json").write_text(json.dumps(setup, indent=2))
    while not (out / "release_shutdown").exists():   # the orchestrator releases after the fault tests
        time.sleep(0.1)
    send_packet(sock, {"type": "SHUTDOWN"})
    sock.close()
    return 0


# ----------------------------------------------------------------------------------------------- sites
def site_worker(args, site_id, q):
    import openfhe as fhe  # noqa: F401  (each site is its own process with its own OpenFHE state)
    from encmonitor.he import serde
    from encmonitor.he.context import pad_vector

    R, _, _ = load_windows(args)
    log = FrameLog(Path(args.out_dir) / f"frames_site{site_id:02d}.jsonl", f"site{site_id}")
    sock = connect(args.host, args.port)
    t0 = time.perf_counter()
    send_packet(sock, {"type": "GET_PUBLIC", "site_id": site_id})
    h1, ctx_blob, f1 = recv_packet(sock)
    h2, pk_blob, f2 = recv_packet(sock)
    log.add("recv", h1, f1); log.add("recv", h2, f2)
    cc = serde.deserialize_context(ctx_blob)
    pk = deserde_public_key(pk_blob)
    t_setup = time.perf_counter() - t0
    profile_id, batch = h1["profile_id"], h1["batch_size"]
    timings = []
    for w in range(R.shape[0]):
        t0 = time.perf_counter()
        pt = cc.MakeCKKSPackedPlaintext(pad_vector(R[w, site_id], batch).tolist())
        pt.SetLength(batch)
        ct = cc.Encrypt(pk, pt)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        blob = serde.serialize_ciphertext(ct)
        t_ser = time.perf_counter() - t0
        t0 = time.perf_counter()
        hdr = {"type": "UPLOAD_RESIDUAL", "run_id": args.run_id, "window_id": w, "site_id": site_id,
               "m": int(R.shape[2]), "sample_count": 250, "profile_id": profile_id}
        log.add("send", hdr, send_packet(sock, hdr, blob))
        ack, _, _ = recv_packet(sock)
        t_up = time.perf_counter() - t0
        if ack["type"] != "ACK":
            q.put({"site": site_id, "error": ack})
            return
        timings.append({"window": w, "encrypt_sec": t_enc, "serialize_sec": t_ser, "upload_ack_sec": t_up, "bytes": len(blob)})
    sock.close()
    q.put({"site": site_id, "setup_sec": t_setup, "timings": timings, "profile_id": profile_id})


def run_sites(args):
    import multiprocessing as mp
    R, _, _ = load_windows(args)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=site_worker, args=(args, s, q)) for s in range(R.shape[1])]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    (Path(args.out_dir) / "sites_results.json").write_text(json.dumps(results))
    return 0 if all("error" not in r for r in results) else 1


# ----------------------------------------------------------------------------------------------- fault tests
def run_faults(args):
    """Server hard errors, tested against a live server with a real session context."""
    from encmonitor.he import serde
    from encmonitor.he.context import pad_vector

    sock = connect(args.host, args.port)
    send_packet(sock, {"type": "GET_PUBLIC", "site_id": 0})
    h1, ctx_blob, _ = recv_packet(sock)
    _, pk_blob, _ = recv_packet(sock)
    cc, pk = serde.deserialize_context(ctx_blob), deserde_public_key(pk_blob)
    pt = cc.MakeCKKSPackedPlaintext(pad_vector(np.zeros(h1["m"]), h1["batch_size"]).tolist())
    blob = serde.serialize_ciphertext(cc.Encrypt(pk, pt))
    wid, S = 10_000_000, args.sites
    send_packet(sock, {"type": "REGISTER_WINDOW", "run_id": "fault", "window_id": wid, "expected_sites": S,
                       "weights": [1.0 / S] * S, "weights_hash": "x"}); recv_packet(sock)
    base = {"type": "UPLOAD_RESIDUAL", "run_id": "fault", "window_id": wid, "site_id": 0, "m": h1["m"],
            "sample_count": 250, "profile_id": h1["profile_id"]}
    cases = [("first upload accepted", base, "ACK"),
             ("duplicate upload", base, "DUPLICATE_UPLOAD"),
             ("wrong dimension", {**base, "site_id": 1, "m": h1["m"] + 1}, "DIMENSION_MISMATCH"),
             ("wrong profile/context", {**base, "site_id": 1, "profile_id": "0000000000000000"}, "PROFILE_MISMATCH"),
             ("unknown window", {**base, "site_id": 1, "window_id": wid + 1}, "UNKNOWN_WINDOW"),
             ("site outside set", {**base, "site_id": S}, "UNKNOWN_SITE")]
    out = []
    for name, hdr, expect in cases:
        send_packet(sock, hdr, blob)
        r, _, _ = recv_packet(sock)
        got = r["type"] if r["type"] == "ACK" else r.get("code")
        out.append({"case": name, "expected": expect, "got": got, "ok": got == expect})
    send_packet(sock, {"type": "COMPUTE", "run_id": "fault", "window_id": wid, "timeout_sec": 2})
    r, _, _ = recv_packet(sock)
    out.append({"case": "missing sites -> timeout, window FAILED, no renormalization", "expected": "TIMEOUT",
                "got": r.get("code"), "ok": r.get("code") == "TIMEOUT", "detail": r.get("detail")})
    sock.close()
    (Path(args.out_dir) / "fault_tests.json").write_text(json.dumps(out, indent=2))
    return 0


# ----------------------------------------------------------------------------------------------- orchestrator
def summarize(args):
    out = Path(args.out_dir)
    frames = []
    for f in out.glob("frames_*.jsonl"):
        frames += [json.loads(l) for l in open(f)]
    by_type = {}
    for fr in frames:
        if fr["direction"] != "send":
            continue
        d = by_type.setdefault(fr["type"], {"frames": 0, "wire_bytes_total": 0, "blob_bytes_max": 0})
        d["frames"] += 1
        d["wire_bytes_total"] += fr["wire_bytes"]
        d["blob_bytes_max"] = max(d["blob_bytes_max"], fr["blob_bytes"])
    res = [json.loads(l) for l in open(out / "authority_results.jsonl")]
    sites = json.load(open(out / "sites_results.json"))
    setup = json.load(open(out / "authority_setup.json"))
    faults = json.load(open(out / "fault_tests.json")) if (out / "fault_tests.json").exists() else []
    N, S = len(res), len(sites)
    q = lambda v, p: float(np.quantile(v, p))
    enc = [t["encrypt_sec"] for s in sites for t in s["timings"]]
    up = [t["upload_ack_sec"] for s in sites for t in s["timings"]]
    rt = [r["timing_sec"]["compute_round_trip"] for r in res]
    comp = [r["timing_sec"]["server_compute"] for r in res]
    dec = [r["timing_sec"]["authority_decrypt"] for r in res]
    upload_bytes = by_type.get("UPLOAD_RESIDUAL", {}).get("wire_bytes_total", 0) / max(N, 1)
    result_bytes = by_type.get("RESULT", {}).get("wire_bytes_total", 0) / max(N, 1)
    online_bytes = upload_bytes + result_bytes
    profiles = {"1ms_1Gbps": (0.001, 1e9), "20ms_100Mbps": (0.020, 100e6), "80ms_20Mbps": (0.080, 20e6)}
    # Model (not netem): per window, sites upload in parallel (one ciphertext each) then the result frames return.
    modeled = {name: {"upload_sec": lat + (upload_bytes / S) * 8 / bw, "results_sec": lat + result_bytes * 8 / bw,
                      "network_total_sec": 2 * lat + (upload_bytes / S + result_bytes) * 8 / bw}
               for name, (lat, bw) in profiles.items()}
    diffs = [r["max_abs_diff_vs_inprocess_ckks"] for r in res if "max_abs_diff_vs_inprocess_ckks" in r]
    one_context = len({s.get("profile_id") for s in sites}) == 1 and all(len(s["timings"]) == N for s in sites)
    checks = {
        "all_S_ciphertexts_accepted_under_one_context": bool(one_context and by_type.get("UPLOAD_RESIDUAL", {}).get("frames") == N * S),
        "outputs_match_plaintext_abs_err_lt_1e-3": max(r["max_abs_err_vs_plaintext"] for r in res) < 1e-3,
        "outputs_match_inprocess_ckks": (max(diffs) < 1e-9) if diffs else None,
        "decisions_agree_with_plaintext": all(r.get("decisions_agree_with_plaintext", True) for r in res),
        "top1_agrees_with_plaintext": all(r["top1_same_as_plaintext"] for r in res),
        "fault_tests_all_rejected_correctly": all(f["ok"] for f in faults) if faults else None,
    }
    summary = {"label": args.label, "windows": N, "sites": S, "m": None,
               "setup": setup, "frames_by_type": by_type,
               "per_window": {"upload_wire_bytes_all_sites": upload_bytes, "result_wire_bytes": result_bytes,
                              "online_wire_bytes": online_bytes,
                              "site_encrypt_sec_p50_p95": [q(enc, .5), q(enc, .95)],
                              "site_upload_ack_sec_p50_p95": [q(up, .5), q(up, .95)],
                              "server_compute_sec_p50_p95": [q(comp, .5), q(comp, .95)],
                              "compute_round_trip_sec_p50_p95": [q(rt, .5), q(rt, .95)],
                              "authority_decrypt_sec_p50_p95": [q(dec, .5), q(dec, .95)]},
               "max_abs_err_vs_plaintext": max(r["max_abs_err_vs_plaintext"] for r in res),
               "max_abs_diff_vs_inprocess_ckks": max(diffs) if diffs else None,
               "modeled_network_profiles": modeled, "fault_tests": faults, "checks": checks,
               "network_pass": all(v for v in checks.values() if v is not None)}
    summary["m"] = json.load(open(out / "authority_ready"))["m"]
    (out / "session_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_session(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("*")):
        if f.is_file():
            f.unlink()
    me = [sys.executable, str(Path(__file__).resolve())]
    common = ["--host", args.host, "--port", str(args.port), "--out-dir", str(out), "--run-id", args.run_id,
              "--windows", str(args.windows), "--sites", str(args.sites), "--m", str(args.m), "--timeout", str(args.timeout)]
    if args.replay:
        common += ["--replay", args.replay]
    if args.ckks_jsonl:
        common += ["--ckks-jsonl", args.ckks_jsonl]
    t_start = time.perf_counter()
    server = subprocess.Popen(me + ["server"] + common) if not args.remote_server else None
    while server and not (out / "server_ready").exists():
        time.sleep(0.05)
    authority = subprocess.Popen(me + ["authority"] + common)
    while not (out / "authority_ready").exists():
        if authority.poll() is not None:
            raise RuntimeError("authority exited early")
        time.sleep(0.05)
    sites = subprocess.run(me + ["sites"] + common)
    if sites.returncode != 0:
        authority.kill()
        raise RuntimeError("a site upload was rejected; see sites_results.json")
    if args.faults:
        subprocess.run(me + ["faults"] + common, check=True)
    (out / "release_shutdown").write_text("1")
    authority.wait()
    if server:
        server.wait(timeout=60)
    s = summarize(args)
    s["session_wall_sec"] = time.perf_counter() - t_start
    (out / "session_summary.json").write_text(json.dumps(s, indent=2))
    pw = s["per_window"]
    print(f"[{args.label}] S={s['sites']} m={s['m']} windows={s['windows']}  keys: gen {s['setup']['keygen_sec']:.2f}s, "
          f"distribute {s['setup']['distribute_keys_sec']:.2f}s  wall {s['session_wall_sec']:.1f}s")
    print(f"   per window: upload {pw['upload_wire_bytes_all_sites']/1e6:.2f} MB, results {pw['result_wire_bytes']/1e6:.2f} MB; "
          f"site encrypt p50 {pw['site_encrypt_sec_p50_p95'][0]*1e3:.1f} ms; server compute p50/p95 "
          f"{pw['server_compute_sec_p50_p95'][0]:.3f}/{pw['server_compute_sec_p50_p95'][1]:.3f} s; round trip p50/p95 "
          f"{pw['compute_round_trip_sec_p50_p95'][0]:.3f}/{pw['compute_round_trip_sec_p50_p95'][1]:.3f} s")
    d = s["max_abs_diff_vs_inprocess_ckks"]
    print(f"   max |net - plaintext| = {s['max_abs_err_vs_plaintext']:.2e}   max |net - in-process CKKS| = "
          f"{'n/a' if d is None else f'{d:.2e}'}")
    if s["fault_tests"]:
        print("   fault tests: " + ", ".join(f"{f['case']}: {'OK' if f['ok'] else 'FAIL ('+str(f['got'])+')'}" for f in s["fault_tests"]))
    print(f"   checks: {json.dumps(s['checks'])}  ->  network {'PASS' if s['network_pass'] else 'CHECK'}")
    return 0 if s["network_pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="EncMonitor networked prototype")
    ap.add_argument("role", choices=["run", "server", "authority", "sites", "faults"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50555)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="g5")
    ap.add_argument("--label", default="session")
    ap.add_argument("--replay", default="", help="CKKS replay NPZ (real residual windows); otherwise synthetic windows")
    ap.add_argument("--ckks-jsonl", default="", help="In-process CKKS results for the same windows (comparison)")
    ap.add_argument("--windows", type=int, default=50)
    ap.add_argument("--sites", type=int, default=5, help="synthetic sessions only")
    ap.add_argument("--m", type=int, default=64, help="synthetic sessions only")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--faults", action="store_true", help="run the §7.6 hard-error tests against the live server")
    ap.add_argument("--remote-server", action="store_true", help="server already running on --host (e.g. another machine)")
    args = ap.parse_args()
    return {"run": run_session, "server": run_server, "authority": run_authority, "sites": run_sites,
            "faults": run_faults}[args.role](args)


if __name__ == "__main__":
    raise SystemExit(main())
