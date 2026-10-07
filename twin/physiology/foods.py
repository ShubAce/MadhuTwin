"""Indian food table used by the simulator and the what-if engine.

Values are approximate per-serving estimates assembled from the Indian Food Composition
Tables (IFCT 2017, National Institute of Nutrition, Hyderabad) and published glycaemic-index
studies of Indian foods. They are intended for simulation, not dietary prescription.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Food:
    key: str
    name: str
    serving: str
    carbs: float    # g
    protein: float  # g
    fat: float      # g
    fiber: float    # g
    gi: float       # glycaemic index (glucose = 100)
    region: str = "pan-India"
    tags: tuple[str, ...] = ()

    @property
    def kcal(self) -> float:
        return 4 * self.carbs + 4 * self.protein + 9 * self.fat

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kcal"] = round(self.kcal)
        d["tags"] = list(self.tags)
        return d


_F = Food
FOODS: dict[str, Food] = {f.key: f for f in [
    # Breakfast / tiffin
    _F("idli_sambar", "Idli (3) with sambar", "3 idli + 1 bowl sambar", 58, 11, 4, 6, 69, "south", ("breakfast", "veg")),
    _F("plain_dosa", "Plain dosa with chutney", "1 dosa + chutney", 40, 6, 12, 3, 70, "south", ("breakfast", "veg")),
    _F("masala_dosa", "Masala dosa", "1 dosa + sambar + chutney", 68, 10, 18, 6, 72, "south", ("breakfast", "veg")),
    _F("upma", "Rava upma", "1 cup", 35, 5, 7, 2, 66, "south", ("breakfast", "veg")),
    _F("pongal", "Ven pongal", "1 cup", 45, 7, 10, 2, 66, "south", ("breakfast", "veg")),
    _F("poha", "Kanda poha", "1 plate", 42, 5, 7, 2, 65, "west", ("breakfast", "veg")),
    _F("aloo_paratha", "Aloo paratha with curd", "2 parathas + curd", 70, 12, 22, 6, 64, "north", ("breakfast", "veg")),
    _F("puri_bhaji", "Puri bhaji", "3 puri + aloo bhaji", 62, 8, 24, 5, 70, "north", ("breakfast", "veg")),
    _F("bread_omelette", "Bread omelette", "2 slices + 2 eggs", 27, 16, 16, 2, 70, "pan-India", ("breakfast", "egg")),
    _F("oats", "Oats porridge with milk", "1 bowl", 32, 9, 6, 4, 55, "pan-India", ("breakfast", "veg", "low-gi")),
    _F("moong_chilla", "Moong dal chilla", "2 chillas", 28, 14, 7, 6, 38, "north", ("breakfast", "veg", "low-gi")),
    _F("ragi_dosa", "Ragi dosa", "2 dosas", 38, 5, 6, 6, 60, "south", ("breakfast", "veg", "millet")),
    # Main meals
    _F("rice_sambar", "Rice with sambar and poriyal", "1.5 cups rice + sambar + veg", 85, 12, 9, 8, 72, "south", ("lunch", "dinner", "veg")),
    _F("curd_rice", "Curd rice", "1.5 cups", 52, 9, 7, 1, 60, "south", ("lunch", "dinner", "veg")),
    _F("rice_dal", "Rice and dal", "1.5 cups rice + 1 bowl dal", 88, 15, 6, 7, 68, "pan-India", ("lunch", "dinner", "veg")),
    _F("roti_sabzi", "Roti (3) with sabzi and dal", "3 roti + sabzi + dal", 78, 18, 14, 13, 55, "north", ("lunch", "dinner", "veg")),
    _F("rajma_chawal", "Rajma chawal", "1 cup rice + 1 bowl rajma", 80, 16, 8, 10, 52, "north", ("lunch", "dinner", "veg")),
    _F("chole_bhature", "Chole bhature", "2 bhature + chole", 90, 16, 32, 10, 68, "north", ("lunch", "veg")),
    _F("chicken_biryani", "Chicken biryani with raita", "1 plate", 82, 28, 22, 3, 70, "pan-India", ("lunch", "dinner", "non-veg")),
    _F("veg_biryani", "Veg biryani with raita", "1 plate", 78, 12, 16, 5, 70, "pan-India", ("lunch", "dinner", "veg")),
    _F("fish_curry_rice", "Fish curry rice", "1.5 cups rice + curry", 72, 26, 12, 2, 70, "east", ("lunch", "dinner", "non-veg")),
    _F("ragi_mudde", "Ragi mudde with saaru", "2 balls + saaru", 70, 9, 4, 11, 65, "south", ("lunch", "dinner", "veg", "millet")),
    _F("jowar_roti_meal", "Jowar roti with palya", "2 roti + palya + dal", 58, 14, 10, 12, 52, "south", ("lunch", "dinner", "veg", "millet", "low-gi")),
    _F("khichdi", "Moong dal khichdi", "1.5 cups", 50, 12, 6, 6, 52, "pan-India", ("dinner", "veg", "low-gi")),
    _F("paneer_roti", "Paneer curry with roti", "2 roti + paneer", 45, 22, 24, 6, 45, "north", ("dinner", "veg")),
    _F("chicken_curry_roti", "Chicken curry with roti", "2 roti + curry", 42, 30, 20, 5, 45, "north", ("dinner", "non-veg")),
    _F("brown_rice_dal", "Brown rice, dal and salad", "1 cup brown rice + dal + salad", 62, 15, 6, 10, 50, "pan-India", ("lunch", "dinner", "veg", "low-gi")),
    _F("pav_bhaji", "Pav bhaji", "2 pav + bhaji", 66, 11, 20, 7, 70, "west", ("dinner", "veg")),
    # Snacks and beverages
    _F("chai_sugar", "Masala chai with sugar", "1 cup", 13, 3, 3, 0, 62, "pan-India", ("snack", "beverage")),
    _F("chai_nosugar", "Chai without sugar", "1 cup", 5, 3, 3, 0, 30, "pan-India", ("snack", "beverage", "low-gi")),
    _F("filter_coffee", "Filter coffee with sugar", "1 tumbler", 12, 3, 3, 0, 60, "south", ("snack", "beverage")),
    _F("biscuits", "Marie biscuits", "4 biscuits", 22, 2, 4, 1, 66, "pan-India", ("snack",)),
    _F("samosa", "Samosa", "1 piece", 26, 4, 16, 2, 62, "north", ("snack", "veg")),
    _F("medu_vada", "Medu vada with chutney", "2 vadas", 32, 9, 17, 3, 60, "south", ("snack", "veg")),
    _F("banana", "Banana", "1 medium", 27, 1, 0, 3, 51, "pan-India", ("snack", "fruit")),
    _F("apple", "Apple", "1 medium", 25, 0, 0, 4, 36, "pan-India", ("snack", "fruit", "low-gi")),
    _F("sprouts", "Sprouts chaat", "1 bowl", 18, 9, 2, 7, 32, "pan-India", ("snack", "veg", "low-gi")),
    _F("roasted_chana", "Roasted chana", "1 handful", 16, 8, 2, 6, 28, "pan-India", ("snack", "veg", "low-gi")),
    _F("gulab_jamun", "Gulab jamun", "2 pieces", 46, 4, 12, 0, 76, "pan-India", ("snack", "sweet", "festival")),
    _F("jalebi", "Jalebi", "3 pieces", 48, 2, 10, 0, 80, "north", ("snack", "sweet", "festival")),
    _F("laddoo", "Besan laddoo", "2 pieces", 34, 6, 14, 2, 60, "pan-India", ("snack", "sweet", "festival")),
]}


def by_tag(tag: str) -> list[Food]:
    return [f for f in FOODS.values() if tag in f.tags]


def absorption_time_constant(carbs: float, fat: float, fiber: float, protein: float, gi: float) -> float:
    """Gut absorption time constant (min) for a mixed meal.

    High-GI carbohydrates absorb quickly; fat, fibre and protein slow gastric emptying.
    Calibrated so pure glucose (GI 100) peaks ~20-25 min and a low-GI mixed meal ~60-75 min.
    """
    tau = 22.0 + 0.55 * (100.0 - gi)
    if carbs > 0:
        tau *= 1.0 + 0.6 * min(fat / carbs, 1.0) + 0.8 * min(fiber / carbs, 0.5) + 0.2 * min(protein / carbs, 1.0)
    return float(min(max(tau, 18.0), 120.0))
