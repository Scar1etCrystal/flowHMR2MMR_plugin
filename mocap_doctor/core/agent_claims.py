"""Advisory scope leases + write journal for multi-agent co-editing (pure Python).

Why: with 3–5 agents writing through one socket server, a single global data
version changes on every write, so ``expect_version`` makes agents reject each
other's unrelated work and loop on re-reads.  Concurrency is narrowed to SCOPE
(bones × inclusive frame range):

- ``LeaseTable``: an agent *claims* the bones/frames it is about to edit.  A
  write that hits another agent's live claim on the same bone with overlapping
  frames is refused (E_CLAIMED).  Parent/child overlaps (e.g. forearm vs hand)
  are reported as warnings - the parent's edit moves the child in world space -
  and refused only with ``strict``.  Leases expire (TTL) so a crashed agent
  never blocks the rest; every claim/write by the agent renews its leases.
  Advisory, not a lock: the GUI user is never blocked.
- ``WriteJournal``: every effective write records (version, owner, bones,
  frames).  A stale ``expect_version`` is only an error when something written
  since then by SOMEONE ELSE touches the caller's scope (same bone or an
  ancestor of it, overlapping frames) - or when the change is unscoped
  (external edit, e.g. the user keyed something in the GUI).

No bpy import: the bridge passes ``ancestors(bone) -> set`` for hierarchy
questions, so this module is unit-testable in plain Python.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, Iterable

ALL = None            # bones=None means "every bone"
DEFAULT_TTL_S = 900.0


def frames_overlap(f1, f2) -> bool:
    if f1 is None or f2 is None:
        return True
    return max(int(f1[0]), int(f2[0])) <= min(int(f1[1]), int(f2[1]))


def frames_intersection(f1, f2):
    if f1 is None:
        return list(f2) if f2 is not None else None
    if f2 is None:
        return list(f1)
    return [max(int(f1[0]), int(f2[0])), min(int(f1[1]), int(f2[1]))]


def _bones_hard(b1, b2) -> set | None:
    """Shared bones (None = 'everything')."""
    if b1 is ALL and b2 is ALL:
        return ALL
    if b1 is ALL:
        return set(b2)
    if b2 is ALL:
        return set(b1)
    return set(b1) & set(b2)


def _bones_related(b1, b2, ancestors: Callable[[str], set]) -> list:
    """(ancestor, descendant) pairs across the two sets."""
    if b1 is ALL or b2 is ALL:
        return []
    pairs = []
    for x in b1:
        anc = ancestors(x)
        for y in b2:
            if y in anc:
                pairs.append((y, x))
    for y in b2:
        anc = ancestors(y)
        for x in b1:
            if x in anc:
                pairs.append((x, y))
    return pairs


class LeaseTable:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._claims: dict[str, dict] = {}
        self._seq = itertools.count(1)

    # -- housekeeping ---------------------------------------------------
    def prune(self) -> list:
        now = self._clock()
        dead = [cid for cid, c in self._claims.items() if c["expires"] <= now]
        for cid in dead:
            self._claims.pop(cid, None)
        return dead

    def renew(self, agent_id: str) -> int:
        now = self._clock()
        n = 0
        for c in self._claims.values():
            if c["agent_id"] == agent_id:
                c["expires"] = now + c["ttl_s"]
                n += 1
        return n

    def clear(self):
        self._claims.clear()

    # -- queries --------------------------------------------------------
    def conflicts(self, agent_id, bones, frames,
                  ancestors: Callable[[str], set]):
        """(hard, soft) lists of conflict rows against OTHER agents' claims."""
        self.prune()
        hard, soft = [], []
        for c in self._claims.values():
            if c["agent_id"] == agent_id:
                continue
            if not frames_overlap(frames, c["frames"]):
                continue
            shared = _bones_hard(bones, c["bones"])
            fr = frames_intersection(frames, c["frames"])
            if shared is ALL or shared:
                hard.append({"agent_id": c["agent_id"], "claim_id": c["id"],
                             "bones": "ALL" if shared is ALL else sorted(shared),
                             "frames": fr, "note": c["note"],
                             "ttl_left_s": round(c["expires"] - self._clock(), 1)})
                continue
            pairs = _bones_related(bones, c["bones"], ancestors)
            if pairs:
                soft.append({"agent_id": c["agent_id"], "claim_id": c["id"],
                             "pairs": [f"{a}→{d}" for a, d in pairs[:6]],
                             "frames": fr, "note": c["note"]})
        return hard, soft

    def covering(self, agent_id, bones, frames) -> bool:
        """Does the agent already hold claims covering every bone × frame?"""
        self.prune()
        mine = [c for c in self._claims.values() if c["agent_id"] == agent_id]
        if not mine:
            return False
        a, b = (None, None) if frames is None else (int(frames[0]), int(frames[1]))
        check_bones = [ALL] if bones is ALL else list(bones)
        for bone in check_bones:
            spans = sorted(
                (c["frames"] if c["frames"] is not None else (-10**9, 10**9))
                for c in mine
                if c["bones"] is ALL or (bone is not ALL and bone in c["bones"]))
            if not spans:
                return False
            if a is None:
                if spans[0] != (-10**9, 10**9):
                    return False
                continue
            cur = a
            for s0, s1 in spans:      # interval cover of [a, b]
                if s0 > cur:
                    break
                cur = max(cur, int(s1) + 1)
                if cur > b:
                    break
            if cur <= b:
                return False
        return True

    def table(self) -> list:
        self.prune()
        now = self._clock()
        return [{"claim_id": c["id"], "agent_id": c["agent_id"],
                 "bones": "ALL" if c["bones"] is ALL else sorted(c["bones"]),
                 "frames": list(c["frames"]) if c["frames"] is not None else "ALL",
                 "note": c["note"], "auto": c["auto"],
                 "ttl_left_s": round(c["expires"] - now, 1)}
                for c in sorted(self._claims.values(), key=lambda c: c["id"])]

    def of(self, agent_id) -> list:
        return [r for r in self.table() if r["agent_id"] == agent_id]

    # -- mutations ------------------------------------------------------
    def claim(self, agent_id: str, bones, frames, *,
              ancestors: Callable[[str], set], ttl_s: float = DEFAULT_TTL_S,
              note: str = "", strict: bool = False, check_only: bool = False,
              auto: bool = False) -> dict:
        if not agent_id:
            raise ValueError("agent_id 必填")
        bones = ALL if bones is ALL else frozenset(bones)
        frames = None if frames is None else (int(frames[0]), int(frames[1]))
        if frames is not None and frames[0] > frames[1]:
            raise ValueError(f"frames 起点大于终点：{list(frames)}")
        hard, soft = self.conflicts(agent_id, bones, frames, ancestors)
        granted = not hard and not (strict and soft)
        out = {"granted": granted, "conflicts": hard, "related": soft,
               "claim_id": None, "check_only": bool(check_only)}
        if not granted or check_only:
            return out
        now = self._clock()
        for c in self._claims.values():           # same scope again = renew
            if c["agent_id"] == agent_id and c["bones"] == bones \
                    and c["frames"] == frames:
                c["expires"] = now + float(ttl_s)
                c["ttl_s"] = float(ttl_s)
                if note:
                    c["note"] = note
                out["claim_id"] = c["id"]
                out["renewed"] = True
                self.renew(agent_id)
                return out
        cid = f"c{next(self._seq)}"
        self._claims[cid] = {"id": cid, "agent_id": str(agent_id),
                             "bones": bones, "frames": frames,
                             "note": str(note or ""), "auto": bool(auto),
                             "ttl_s": float(ttl_s), "created": now,
                             "expires": now + float(ttl_s)}
        self.renew(agent_id)
        out["claim_id"] = cid
        return out

    def release(self, agent_id: str, claim_id: str | None = None) -> list:
        self.prune()
        if claim_id:
            c = self._claims.get(claim_id)
            if c is None:
                return []
            if c["agent_id"] != agent_id:
                raise PermissionError(
                    f"{claim_id} 属于 {c['agent_id']}，不是 {agent_id}")
            self._claims.pop(claim_id)
            return [claim_id]
        ids = [cid for cid, c in self._claims.items() if c["agent_id"] == agent_id]
        for cid in ids:
            self._claims.pop(cid)
        return ids


class WriteJournal:
    """Who changed what since version v."""

    def __init__(self, maxlen: int = 2000):
        self.maxlen = int(maxlen)
        self.rows: list[dict] = []
        self.floor = 0          # versions <= floor were truncated away

    def note(self, version: int, owner, bones, frames, tool: str = ""):
        self.rows.append({"v": int(version), "owner": owner,
                          "bones": ALL if bones is ALL else frozenset(bones),
                          "frames": None if frames is None
                          else (int(frames[0]), int(frames[1])),
                          "tool": str(tool), "ts": time.time()})
        if len(self.rows) > self.maxlen:
            drop = len(self.rows) - self.maxlen
            self.floor = self.rows[drop - 1]["v"]
            del self.rows[:drop]

    def stale_against(self, expect: int, current: int, agent_id,
                      scope: Iterable, ancestors: Callable[[str], set]):
        """None when `expect` is still good for this scope, else a reason
        dict.  `scope` = [(bones|ALL, frames|None), ...]."""
        if expect == current:
            return None
        if expect > current:
            return {"why": "期望版本比当前还新（服务重启过？场景已重新载入）"}
        if expect < self.floor:
            return {"why": f"v{expect} 太旧，日志已截断（只保留 v{self.floor} 之后）"}
        hits = []
        for row in self.rows:
            if row["v"] <= expect:
                continue
            if row["owner"] is not None and row["owner"] == agent_id:
                continue
            if row["bones"] is ALL and row["frames"] is None:
                hits.append(row)       # unscoped external change
                continue
            for bones, frames in scope:
                if not frames_overlap(frames, row["frames"]):
                    continue
                if row["bones"] is ALL or bones is ALL:
                    hits.append(row)
                    break
                mine = set(bones)
                for b in bones:
                    mine |= set(ancestors(b))
                if mine & set(row["bones"]):
                    hits.append(row)
                    break
        if not hits:
            return None
        return {"why": "期间别人改了与你范围相交的骨/帧（含祖先骨）",
                "changes": [{"v": r["v"], "by": r["owner"] or "external",
                             "tool": r["tool"],
                             "bones": "ALL" if r["bones"] is ALL
                             else sorted(r["bones"])[:8],
                             "frames": list(r["frames"]) if r["frames"] else "ALL"}
                            for r in hits[:8]]}
