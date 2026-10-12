"""Lucky-number engine and the ML + lucky hybrid ticket builder.

Lucky numbers come from the calendar date of the draw, never from past draws:

* Chinese / feng shui: the day pillar (stem element and the element feeding
  it, Luo Shu number elements), the day's zodiac animal, auspicious digits
  (8 prosperity, 6 smooth, 9 long-lasting, with 4 as taboo) and the annual
  flying star.
* Numerology: weekday planet number, sun-sign ruling planet number, the date's
  universal-day number, day of month and an approximate moon-age day.
* Personal: optional birth date numbers and any numbers the user collected
  elsewhere (horoscope sites, fortune cookies, dreams, plates, ...).

Every number in the pool gets a score and the reasons behind it. A hybrid
ticket takes ``ml_count`` numbers from the ML scores and fills the rest with
lucky numbers; ticket #1 uses the strongest of each, later tickets sample.
The same builder runs in the Strategy Arena, so it is scored against history.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import numpy as np

from .games import GameConfig

STEMS = ("Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui")
STEM_ELEMENT = ("Wood", "Wood", "Fire", "Fire", "Earth", "Earth",
                "Metal", "Metal", "Water", "Water")
BRANCHES = ("Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai")
ANIMALS = ("Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
           "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig")
# Traditional lucky digits per zodiac animal.
ANIMAL_DIGITS = {
    "Rat": (2, 3), "Ox": (1, 4), "Tiger": (1, 3, 4), "Rabbit": (3, 4, 6),
    "Dragon": (1, 6, 7), "Snake": (2, 8, 9), "Horse": (2, 3, 7), "Goat": (3, 4, 9),
    "Monkey": (1, 7, 8), "Rooster": (5, 7, 8), "Dog": (3, 4, 9), "Pig": (2, 5, 8),
}
# Luo Shu (feng shui) element of each single digit; 0 / 10 count as Earth.
ROOT_ELEMENT = {1: "Water", 2: "Earth", 3: "Wood", 4: "Wood", 5: "Earth",
                6: "Metal", 7: "Metal", 8: "Earth", 9: "Fire"}
PRODUCES = {"Wood": "Fire", "Fire": "Earth", "Earth": "Metal", "Metal": "Water", "Water": "Wood"}
MOTHER = {child: parent for parent, child in PRODUCES.items()}
# He Tu: the numbers (last digits) that belong to each element.
HE_TU_DIGITS = {"Water": (1, 6), "Fire": (2, 7), "Wood": (3, 8), "Metal": (4, 9), "Earth": (5, 0)}
# Auspicious digits by sound: 8 fa/prosper, 6 liu/smooth, 9 jiu/lasting, 2 easy, 3 life, 1 first.
# These never change with the date, so they only break ties between numbers the date already favours.
AUSPICIOUS_DIGITS = {8: 0.3, 6: 0.2, 9: 0.2, 2: 0.1, 3: 0.1, 1: 0.05}
AUSPICIOUS_CAP = 0.5
YEAR_STAR_WEIGHT = 0.25  # the flying star is constant for a whole Chinese year
TABOO_FACTOR = 0.6  # per digit 4 ("death" sound), applied to the Chinese score only
# Chaldean / Vedic planet numbers: Sun 1, Moon 2, Jupiter 3, Mercury 5, Venus 6, Saturn 8, Mars 9.
WEEKDAY_NUMBER = (2, 9, 5, 3, 6, 8, 1)  # Monday .. Sunday
SIGNS = (  # (month, first day, sign, ruling-planet number)
    (1, 20, "Aquarius", 8), (2, 19, "Pisces", 3), (3, 21, "Aries", 9), (4, 20, "Taurus", 6),
    (5, 21, "Gemini", 5), (6, 21, "Cancer", 2), (7, 23, "Leo", 1), (8, 23, "Virgo", 5),
    (9, 23, "Libra", 6), (10, 23, "Scorpio", 9), (11, 22, "Sagittarius", 3), (12, 22, "Capricorn", 8),
)
HYBRID_MODE = "Hybrid: ML + Lucky numbers"  # dashboard selection-method label
SYNODIC_MONTH = 29.530588853
NEW_MOON_JD = 2451550.26  # 2000-01-06 18:14 UTC
BASE_WEIGHT = 0.3  # floor added to every score before sampling
EXTRA_NUMBER_WEIGHT = 2.0
AI_NUMBER_WEIGHT = 1.5  # numbers the AI suggested for the draw date
OFF_SHORTLIST_FACTOR = 1e-9  # with an AI shortlist, other numbers only fill gaps (ranked by the date rules)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def digit_root(n: int) -> int:
    return 0 if n == 0 else 1 + (n - 1) % 9


def _digits(n: int) -> set[int]:
    return {int(c) for c in str(n)}


def _julian_day(d: dt.date) -> int:
    return d.toordinal() + 1721425


def _chinese_year(d: dt.date) -> int:
    """Chinese solar year, which turns over at Li Chun (about 4 February)."""
    return d.year if (d.month, d.day) >= (2, 4) else d.year - 1


def day_profile(d: dt.date) -> dict:
    """Calendar facts every lucky-number source works from."""
    index = (_julian_day(d) + 49) % 60  # 0 = Jia-Zi
    stem, branch = index % 10, index % 12
    year = _chinese_year(d)
    element = STEM_ELEMENT[stem]
    sign = next((s for m, day, s, _ in reversed(SIGNS) if (d.month, d.day) >= (m, day)), "Capricorn")
    sign_number = next((n for _, _, s, n in SIGNS if s == sign), 8)
    moon_age = (_julian_day(d) - NEW_MOON_JD) % SYNODIC_MONTH
    return {
        "date": d.isoformat(), "weekday": _WEEKDAYS[d.weekday()],
        "day_pillar": f"{STEMS[stem]}-{BRANCHES[branch]}",
        "day_stem_number": stem + 1, "day_branch_number": branch + 1,
        "day_element": element, "day_mother_element": MOTHER[element],
        "day_animal": ANIMALS[branch],
        "chinese_year": year, "year_pillar": f"{STEMS[(year - 4) % 10]}-{BRANCHES[(year - 4) % 12]}",
        "year_animal": ANIMALS[(year - 4) % 12],
        "year_star": 9 - ((year - 2000) % 9),
        "weekday_number": WEEKDAY_NUMBER[d.weekday()],
        "sun_sign": sign, "sign_number": sign_number,
        "universal_day": digit_root(sum(int(c) for c in d.strftime("%Y%m%d"))),
        "moon_age_day": int(moon_age) + 1,
    }


@dataclass
class LuckyScores:
    """Per-number lucky score (pool order) with reasons and the day's profile."""
    numbers: list[int]
    scores: np.ndarray
    reasons: dict[int, list[str]]
    profile: dict
    personal: dict = field(default_factory=dict)
    boosted: np.ndarray | None = None  # AI shortlist (+ own numbers); the lucky half is drawn from it

    def top(self, k: int = 12) -> list[dict]:
        order = sorted(range(len(self.numbers)), key=lambda i: (-self.scores[i], self.numbers[i]))
        return [{"number": self.numbers[i], "score": float(self.scores[i]),
                 "reasons": "; ".join(self.reasons[self.numbers[i]])} for i in order[:k]
                if self.scores[i] > 0]

    def to_dict(self, top: int = 12) -> dict:
        return {"profile": self.profile, "personal": self.personal, "top_numbers": self.top(top),
                "scores": [{"number": n, "score": float(s)} for n, s in zip(self.numbers, self.scores)],
                "reasons": {str(n): list(why) for n, why in self.reasons.items() if why}}


def lucky_scores(cfg: GameConfig, draw_date: dt.date, *, birthdate: dt.date | None = None,
                 extra_numbers: tuple[int, ...] | list[int] = (),
                 ai_numbers: dict[int, str] | None = None) -> LuckyScores:
    """Score every number in the game's pool for a draw on draw_date."""
    prof = day_profile(draw_date)
    numbers = list(range(cfg.low_boundary, cfg.pool + 1))
    chinese, western = np.zeros(len(numbers)), np.zeros(len(numbers))
    personal, extra = np.zeros(len(numbers)), np.zeros(len(numbers))
    reasons: dict[int, list[str]] = {n: [] for n in numbers}
    animal_digits = set(ANIMAL_DIGITS[prof["day_animal"]])
    life_path = digit_root(sum(int(c) for c in birthdate.strftime("%Y%m%d"))) if birthdate else None
    for i, n in enumerate(numbers):
        root, digits = digit_root(n), _digits(n)
        why = reasons[n]
        element = ROOT_ELEMENT.get(root, "Earth")
        if element == prof["day_element"]:
            chinese[i] += 1.0
            why.append(f"{element} day element")
        elif element == prof["day_mother_element"]:
            chinese[i] += 0.6
            why.append(f"{element} feeds the {prof['day_element']} day")
        if n % 10 in HE_TU_DIGITS[prof["day_element"]]:
            chinese[i] += 0.7
            why.append(f"He Tu number of {prof['day_element']}")
        if n == prof["day_stem_number"]:
            chinese[i] += 0.8
            why.append(f"day stem {prof['day_pillar'].split('-')[0]} = {n}")
        if n == prof["day_branch_number"]:
            chinese[i] += 0.8
            why.append(f"day branch {prof['day_pillar'].split('-')[1]} = {n}")
        if digits & animal_digits:
            chinese[i] += 0.8
            why.append(f"{prof['day_animal']} day digit")
        lucky_digits = [c for c in str(n) if int(c) in AUSPICIOUS_DIGITS]
        if lucky_digits:
            chinese[i] += min(sum(AUSPICIOUS_DIGITS[int(c)] for c in lucky_digits), AUSPICIOUS_CAP)
            why.append("auspicious digits " + "".join(lucky_digits))
        if prof["year_star"] in digits:
            chinese[i] += YEAR_STAR_WEIGHT
            why.append(f"flying star {prof['year_star']}")
        fours = str(n).count("4")
        if fours and chinese[i] > 0:
            chinese[i] *= TABOO_FACTOR ** fours
            why.append("contains 4 (taboo)")
        if root == prof["weekday_number"]:
            western[i] += 0.8
            why.append(f"{prof['weekday']} planet number {prof['weekday_number']}")
        if prof["sign_number"] in digits:
            western[i] += 0.4
            why.append(f"{prof['sun_sign']} ruler number {prof['sign_number']}")
        if n == draw_date.day:
            western[i] += 1.0
            why.append("day of month")
        if root == prof["universal_day"]:
            western[i] += 0.7
            why.append(f"universal day {prof['universal_day']}")
        if n == prof["moon_age_day"]:
            western[i] += 1.0
            why.append("moon-age day")
        if birthdate and n in (birthdate.day, birthdate.month):
            personal[i] += 1.0
            why.append("birth day/month")
        if life_path and root == life_path:
            personal[i] += 0.7
            why.append(f"life path {life_path}")
        if n in extra_numbers:
            extra[i] += EXTRA_NUMBER_WEIGHT
            why.append("your own lucky number")
        if ai_numbers and n in ai_numbers:
            extra[i] += AI_NUMBER_WEIGHT
            why.append(f"AI suggestion: {ai_numbers[n]}")
    shortlist = {n for n in (ai_numbers or {}) if n in reasons}
    boosted = (np.array([n in shortlist or n in extra_numbers for n in numbers]) if shortlist else None)
    return LuckyScores(
        numbers, chinese + western + personal + extra, reasons, prof,
        {"birthdate": birthdate.isoformat() if birthdate else None, "life_path": life_path,
         "extra_numbers": [n for n in extra_numbers if n in reasons],
         "ai_numbers": [n for n in (ai_numbers or {}) if n in reasons]},
        boosted,
    )


def parse_extra_numbers(text: str, cfg: GameConfig) -> list[int]:
    """Numbers inside the game's pool from free text, in order, without repeats."""
    seen: dict[int, None] = {}
    for token in re.findall(r"\d+", text or ""):
        value = int(token)
        if cfg.low_boundary <= value <= cfg.pool:
            seen.setdefault(value)
    return list(seen)


def next_draw_date(cfg: GameConfig, now: dt.datetime | None = None) -> dt.date:
    """Next scheduled draw date; today counts until the last slot has started."""
    now = now or dt.datetime.now()
    last_hour = max(int(t[:2]) for t in cfg.draw_times)
    start = 0 if now.hour < last_hour else 1
    for offset in range(start, start + 8):
        day = now.date() + dt.timedelta(days=offset)
        if _WEEKDAYS[day.weekday()] in cfg.draw_days:
            return day
    return now.date()


def ask_ai_lucky_numbers(client, cfg: GameConfig, draw_date: dt.date,
                          birthdate: dt.date | None = None, count: int = 8) -> dict[int, str]:
    """Lucky numbers the AI suggests for draw_date, as {number: reason}.

    The AI gets the locally computed calendar facts (it has no live web access);
    pass the result as ``ai_numbers`` to lucky_scores / hybrid_generation."""
    personal = {}
    if birthdate:
        personal = {"birthdate": birthdate.isoformat(),
                    "life_path": digit_root(sum(int(c) for c in birthdate.strftime("%Y%m%d")))}
    rows = client.find_lucky_numbers(
        cfg.name, cfg.low_boundary, cfg.pool, min(count, cfg.pool - cfg.low_boundary + 1),
        day_profile(draw_date), personal)
    return {int(r["number"]): str(r["reason"]) for r in rows}


def default_ml_count(cfg: GameConfig) -> int:
    return max(1, cfg.picks // 2)


def scores_to_weights(cfg: GameConfig, score_records: list[dict]) -> np.ndarray:
    """ML score records -> positive weights shaped (positions, size)."""
    size = cfg.pool - cfg.low_boundary + 1
    weights = np.ones((cfg.picks if cfg.ordered else 1, size))
    for row in score_records:
        weights[row.get("position", 1) - 1 if cfg.ordered else 0, row["number"] - cfg.low_boundary] = row["score"]
    return weights


def _gumbel_top(rng: np.random.Generator, weights: np.ndarray, k: int, blocked: set[int]) -> list[int]:
    """k indexes sampled without replacement with probability proportional to weights."""
    keys = np.log(np.maximum(weights, 1e-12)) + rng.gumbel(size=len(weights))
    for index in blocked:
        keys[index] = -np.inf
    return [int(i) for i in np.argsort(-keys)[:k]]


def _best(weights: np.ndarray, k: int, blocked: set[int]) -> list[int]:
    """k highest-weight indexes (ties go to the lower number), skipping blocked ones."""
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    return [i for i in order if i not in blocked][:k]


def _lucky_weights(lucky: LuckyScores) -> np.ndarray:
    weights = BASE_WEIGHT + lucky.scores
    if lucky.boosted is not None:  # AI shortlist: draw from it first, the date rules fill any gap
        weights = np.where(lucky.boosted, weights, weights * OFF_SHORTLIST_FACTOR)
    return weights


def build_hybrid_tickets(cfg: GameConfig, ml_weights: np.ndarray, lucky: LuckyScores, n: int,
                         ml_count: int, rng: np.random.Generator) -> list[dict]:
    """Tickets made of ml_count ML-weighted numbers and the rest lucky numbers.

    Ticket #1 takes the strongest ML numbers and the strongest lucky numbers;
    later tickets sample both halves. Unordered tickets never repeat a number.
    Ordered games put the ML digits in the first positions, lucky digits after.
    """
    low = cfg.low_boundary
    ml_count = min(max(int(ml_count), 0), cfg.picks)
    lucky_count = cfg.picks - ml_count
    lucky_w = _lucky_weights(lucky)
    ml_w = np.asarray(ml_weights, dtype=float).reshape(cfg.picks if cfg.ordered else 1, -1)
    tickets: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for attempt in range(max(60 * n, 60)):
        if len(tickets) >= n:
            break
        first = not tickets and attempt == 0
        if cfg.ordered:
            ml_part = [(_best(ml_w[p], 1, set()) if first else _gumbel_top(rng, ml_w[p], 1, set()))[0]
                       for p in range(ml_count)]
            lucky_part = (_best(lucky_w, lucky_count, set()) if first else
                          [_gumbel_top(rng, lucky_w, 1, set())[0] for _ in range(lucky_count)])
            ml_numbers, lucky_numbers = [v + low for v in ml_part], [v + low for v in lucky_part]
            combo = ml_numbers + lucky_numbers
        else:
            ml_part = (_best(ml_w[0], ml_count, set()) if first
                       else _gumbel_top(rng, ml_w[0], ml_count, set()))
            lucky_part = (_best(lucky_w, lucky_count, set(ml_part)) if first
                          else _gumbel_top(rng, lucky_w, lucky_count, set(ml_part)))
            ml_numbers, lucky_numbers = sorted(v + low for v in ml_part), sorted(v + low for v in lucky_part)
            combo = sorted(ml_numbers + lucky_numbers)
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        tickets.append({"numbers": combo, "ml": ml_numbers, "lucky": lucky_numbers})
    return tickets


def lucky_tickets(cfg: GameConfig, lucky: LuckyScores, n: int, rng: np.random.Generator) -> list[list[int]]:
    """Tickets made only of lucky numbers: #1 the strongest, the rest sampled."""
    return [t["numbers"] for t in build_hybrid_tickets(
        cfg, np.ones((cfg.picks if cfg.ordered else 1, len(lucky.numbers))), lucky, n, 0, rng)]


def hybrid_generation(
    df, cfg: GameConfig, *, n: int = 5, window: int | None = None, draw_time: str | None = None,
    draw_date: dt.date | None = None, ml_count: int | None = None, birthdate: dt.date | None = None,
    extra_numbers: tuple[int, ...] | list[int] = (), seed: int = 42,
    bet_feedback: dict[int, float] | None = None, ai_numbers: dict[int, str] | None = None,
) -> dict:
    """ML half from the Lean ML scores, lucky half from the draw date's sources."""
    from .lean_ml import generate_ml_combinations

    draw_date = draw_date or next_draw_date(cfg)
    ml_count = default_ml_count(cfg) if ml_count is None else min(max(int(ml_count), 0), cfg.picks)
    ml = generate_ml_combinations(df, cfg, n=1, window=window, draw_time=draw_time, seed=seed,
                                  bet_feedback=bet_feedback)
    lucky = lucky_scores(cfg, draw_date, birthdate=birthdate, extra_numbers=extra_numbers,
                         ai_numbers=ai_numbers)
    slot = int(draw_time[:2]) if draw_time else 0
    rng = np.random.default_rng([seed, draw_date.toordinal(), slot])
    tickets = build_hybrid_tickets(cfg, scores_to_weights(cfg, ml["scores"]), lucky, n, ml_count, rng)
    result = {
        "method": "hybrid_ml_lucky", "ml_method": ml["method"],
        "lucky_source": "ai_shortlist+date_rules" if ai_numbers else "date_rules",
        "combinations": [t["numbers"] for t in tickets], "tickets": tickets,
        "ml_count": ml_count, "lucky_count": cfg.picks - ml_count, "draw_date": draw_date.isoformat(),
        "lucky": lucky.to_dict(), "scores": ml["scores"], "feature_weights": ml["feature_weights"],
        "evaluation": ml["evaluation"], "training_draws": ml["training_draws"],
        "reason": (f"{ml_count} numbers from ML scores + {cfg.picks - ml_count} lucky numbers for "
                   f"{draw_date.isoformat()}. ML half: {ml['reason']}"),
    }
    if ml.get("bet_feedback_numbers"):  # the ML scores were rescaled by the user's scored bets
        result["bet_feedback_numbers"] = ml["bet_feedback_numbers"]
    return result
