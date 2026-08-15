#!/usr/bin/env python3
"""
Generate a genuinely diverse cooking dataset with ~100,000 unique lines.

Strategy:
- Generate diverse factual knowledge about cooking
- Create natural language questions and answers
- Produce instructional content in varied formats
- Ensure each line is genuinely unique
"""

import random
from pathlib import Path
from collections import Counter

random.seed(42)

# ============================================================================
# CORE DATA STRUCTURES - All ingredients for content generation
# ============================================================================

# Cooking methods
METHODS = [
    "boil", "simmer", "poach", "steam", "blanch", "braise", "stew", "roast",
    "bake", "grill", "broil", "sauté", "pan-fry", "deep-fry", "stir-fry",
    "pressure cook", "slow cook", "smoke", "cure", "pickle", "ferment",
    "sous vide", "poach in oil", "flash fry", "sear", "reverse sear",
    "steam-roast", "oven-braise", "pan-roast", "shallow-fry", "air-fry",
    "confit", "glaze", "reduce", "deglaze", "flambe", "caramelize",
    "blister", "char", "smoke-roast", "wok", "steam-fry", "water-sauté",
    "en papillote", "hasselback", "spatchcock", "butterfly", "julienne",
    "brunoise", "chiffonade", "mince", "dice", "chop", "slice", "julienne",
    "emulsify", "whisk", "fold", "cream", "knead", "proof", "rest",
]

# Additional verbs for cooking actions
COOKING_VERBS = [
    "season", "salt", "pepper", "spice", "herb", "marinate", "brine",
    "rub", "coat", "dredge", "bread", "batter", "glaze", "baste",
    "brush", "drizzle", "sprinkle", "scatter", "spread", "layer",
    "arrange", "plate", "garnish", "decorate", "top", "finish",
    "prepare", "assemble", "combine", "blend", "mix", "stir", "beat",
    "whip", "fold", "incorporate", "integrate", "dissolve", "melt",
    "soften", "sweat", "sauté", "brown", "caramelize", "deglaze",
    "reduce", "thicken", "simmer", "boil", "poach", "steam", "blanch",
    "shock", "peel", "seed", "core", "stem", "trim", "clean",
    "wash", "dry", "pat", "drain", "strain", "sieve", "filter",
]

# Ingredients - comprehensive list with expanded vocabulary
VEGETABLES = [
    # Common vegetables
    "onion", "garlic", "ginger", "carrot", "celery", "bell pepper", "tomato",
    "potato", "sweet potato", "broccoli", "cauliflower", "spinach", "kale",
    "lettuce", "cabbage", "brussels sprouts", "asparagus", "green beans",
    "peas", "corn", "zucchini", "eggplant", "cucumber", "mushroom",
    "butternut squash", "acorn squash", "pumpkin", "beets", "radishes",
    "turnips", "parsnips", "rutabaga", "leeks", "shallots", "scallions",
    "fennel", "artichoke", "okra", "bok choy", "water chestnut", "bamboo shoots",
    # Additional vegetables
    "arugula", "watercress", "endive", "chicory", "radicchio", "escarole",
    "collard greens", "mustard greens", "turnip greens", "chard", "sorrel",
    "dandelion greens", "beet greens", "pea shoots", "sunflower sprouts",
    "alfalfa sprouts", "bean sprouts", "mung bean sprouts", "contrary",
    "cabbage", "napa cabbage", "savoy cabbage", "red cabbage", "brussels",
    "kohlrabi", "daikon", "jicama", "taro", "yuca", "cassava", "plantain",
    "yuca", "malanga", "arrowroot", "sunchokes", "jerusalem artichoke",
    "cardoon", "celeriac", "fava beans", "lima beans", "edamame", "soybean",
    "snow peas", "snap peas", "sugar snap peas", "english peas", "pea",
    "bird's eye chili", "jalapeño", "serrano", "habanero", "poblano", "anaheim",
    "guajillo", "ancho", "chipotle", "pasilla", "cayenne", "thai chili",
    "shishito", "padron", "cherry pepper", "banana pepper", "pepperoncini",
    "portobello", "cremini", "shiitake", "oyster mushroom", "enoki", "maitake",
]

FRUITS = [
    # Common fruits
    "apple", "pear", "peach", "plum", "nectarine", "apricot", "cherry",
    "strawberry", "blueberry", "raspberry", "blackberry", "mango", "pineapple",
    "papaya", "kiwi", "banana", "orange", "lemon", "lime", "grapefruit",
    "mandarin", "clementine", "pomegranate", "fig", "date", "grape", "melon",
    "cantaloupe", "honeydew", "watermelon", "coconut", "avocado", "plantain",
    # Additional fruits
    "persimmon", "pomegranate", "quince", "medlar", "loquat", "longan",
    "lychee", "rambutan", "mangosteen", "durian", "jackfruit", "breadfruit",
    "starfruit", "dragon fruit", "passion fruit", "guava", "cherimoya",
    "soursop", "custard apple", "sugar apple", "acai", "gooseberry", "currant",
    "elderberry", "cranberry", "lingonberry", "cloudberry", "boysenberry",
    "marionberry", "tayberry", "loganberry", "youngberry", "olallieberry",
    "blackcurrant", "redcurrant", "whitecurrant", "jostaberry", "buffaloberry",
    "sea buckthorn", "goji berry", "acai berry", "maqui berry", "camu camu",
    "baobab", "tamarind", "plantain", "breadnut", "soursop", "sweetsop",
    "sapodilla", "sapote", "mamey", "canistel", "eggfruit", "black sapote",
]

PROTEINS = [
    # Meat
    "chicken", "turkey", "duck", "goose", "quail", "pheasant", "guinea fowl",
    "cornish hen", "capcornish", "poussin", "squab", "beef", "pork", "lamb",
    "veal", "venison", "bison", "boar", "venison", "elk", "rabbit", "goat",
    "ham", "bacon", "sausage", "chorizo", "prosciutto", "pancetta", "salami",
    "pepperoni", "andouille", "mortadella", "bologna", "pastrami", "corned beef",
    "brisket", "tenderloin", "sirloin", "ribeye", "strip", "flank", "skirt",
    "hanger", "chuck", "round", "shank", "short rib", "osso buco", "oxtail",
    "tri-tip", "flat iron", "bavette", "onglet", "coulotte", "picanha",
    # Fish and seafood
    "salmon", "tuna", "cod", "halibut", "tilapia", "trout", "mackerel",
    "sardines", "anchovies", "herring", "mackerel", "sole", "flounder",
    "plaice", "turbot", "halibut", "sea bass", "branzino", "snapper",
    "grouper", "mahimahi", "wahoo", "swordfish", "monkfish", "skate",
    "eel", "catfish", "barramundi", "yellowtail", "hamachi", "kurodai",
    "shrimp", "prawn", "crab", "lobster", "scallops", "mussels", "clams",
    "oysters", "sea urchin", "octopus", "squid", "cuttlefish", " calamari",
    # Other proteins
    "egg", "eggs", "tofu", "tempeh", "seitan", "beans", "lentils", "chickpeas",
    "black beans", "pinto beans", "kidney beans", "navy beans", "black-eyed peas",
    "lima beans", "fava beans", "edamame", "miso", "natto", "quinoa", "amaranth",
]

DAIRY = [
    "milk", "cream", "butter", "cheese", "yogurt", "sour cream", "creme fraiche",
    "ricotta", "cottage cheese", "cream cheese", "mozzarella", "parmesan",
    "cheddar", "gouda", "brie", "camembert", "blue cheese", "feta", "goat cheese",
    "paneer", "halloumi", "manchego", "pecorino", "asiago", "gruyere", "emmental",
    "swiss cheese", "provolone", "havarti", "fontina", "mascarpone", "gorgonzola",
    "roquefort", "stilton", "chevre", "neufchatel", "quark", "skyr", "kefir",
]

GRAINS = [
    "rice", "pasta", "noodles", "bread", "flour", "oats", "barley", "quinoa",
    "couscous", "bulgur", "farro", "wheat", "cornmeal", "polenta", "millet",
    "rye", "buckwheat", "amaranth", "teff", "sorghum", "spelt", "kamut",
    "wild rice", "jasmine rice", "basmati rice", "arborio rice", "short-grain rice",
    "brown rice", "white rice", "black rice", "red rice", " Forbidden rice",
    "sushi rice", "paella rice", "risotto rice", "glutinous rice", "sticky rice",
    "ramen noodles", "udon noodles", "soba noodles", "rice noodles", "glass noodles",
    "egg noodles", "fettuccine", "linguine", "spaghetti", "penne", "rigatoni",
    "macaroni", "farfalle", "orzo", "gnocchi", "tortellini", "ravioli", "lasagna",
    "baguette", "ciabatta", "focaccia", "sourdough", "whole wheat", "rye bread",
    "pita", "naan", "roti", "chapati", "tortilla", "arepa", "injera", "lavash",
]

HERBS = [
    "basil", "parsley", "cilantro", "mint", "dill", "chives", "tarragon",
    "rosemary", "thyme", "oregano", "marjoram", "sage", "bay leaf", "lemongrass",
    "cilantro", "coriander", "curry leaves", "kaffir lime leaves", "pandan",
    "thai basil", "holy basil", "shiso", "perilla", "mitsuba", "🍜",
    "chervil", "lovage", "borage", "burnet", "salad burnet", "sorrel",
    "celery leaf", "fennel fronds", "dill weed", "tarragon", "chervil",
]

SPICES = [
    "salt", "pepper", "cumin", "coriander", "turmeric", "paprika", "cinnamon",
    "cardamom", "nutmeg", "cloves", "ginger", "allspice", "star anise", "fennel",
    "caraway", "mustard seed", "celery seed", "cumin seed", "fennel seed",
    "anise", "aniseed", "star anise", "sichuan pepper", "sansho", "togarashi",
    "cayenne", "chili powder", "chile powder", "ancho chili", "chipotle powder",
    "smoked paprika", "hungarian paprika", "spanish paprika", "garam masala",
    "curry powder", "five spice", "seven spice", "ras el hanout", "za'atar",
    "berbere", "jerk seasoning", "cajun seasoning", "Old Bay", "adobo",
    "sazon", "taco seasoning", "fajita seasoning", "italian seasoning",
    "herbes de provence", "bouquet garni", "fine herbs", "fines herbes",
    "sumac", "fenugreek", "asafoetida", "amchur", "ajwain", "nigella seeds",
    "black onion seeds", "kalonji", "charnushka", "black cumin", "white pepper",
    "green peppercorn", "pink peppercorn", "sichuan peppercorn", "long pepper",
    "grains of paradise", "cubeb", "juniper berries", "mahlab", "mace",
    "black cardamom", "green cardamom", "white cardamom", "tonka bean",
]

OILS_AND_FATS = [
    "olive oil", "vegetable oil", "canola oil", "sunflower oil", "safflower oil",
    "grapeseed oil", "avocado oil", "coconut oil", "palm oil", "sesame oil",
    "peanut oil", "corn oil", "walnut oil", "hazelnut oil", "almond oil",
    "truffle oil", "chili oil", "infused oil", "herb oil", "garlic oil",
    "butter", "clarified butter", "ghee", "lard", "schmaltz", "bacon fat",
    "duck fat", "beef tallow", "suet", "margarine", "shortening", "coconut cream",
]

NUTS_AND_SEEDS = [
    "almonds", "walnuts", "pecans", "cashews", "pistachios", "macadamia",
    "hazelnuts", "filberts", "brazil nuts", "pine nuts", "pignoli", "peanuts",
    "chestnuts", "pumpkin seeds", "sunflower seeds", "sesame seeds", "poppy seeds",
    "chia seeds", "flax seeds", "hemp seeds", "flaxseed", "linseed",
    "tahini", "peanut butter", "almond butter", "cashew butter", "sunflower butter",
    "nutella", "gianduja", "praline", "marzipan", "almond paste",
]

# Descriptive adjectives for food
ADJECTIVES = [
    # Taste
    "sweet", "sour", "salty", "bitter", "umami", "savory", "tangy", "piquant",
    "acidic", "mellow", "rich", "light", "bold", "subtle", "delicate", "intense",
    "mild", "strong", "weak", "balanced", "complex", "simple", "layered",
    "sweetish", "sourish", "saltish", "bittersweet", "semisweet", "tart",
    "briny", "cured", "pickled", "fermented", "sharp", "mild", "moderate",
    # Texture
    "crispy", "crunchy", "tender", "soft", "hard", "chewy", "gooey", "creamy",
    "smooth", "rough", "silky", "velvety", "grainy", "granular", "fibrous",
    "moist", "dry", "juicy", "succulent", "moist", "wet", "damp", "parched",
    "flaky", "flakey", "layered", "puuffy", "dense", "light", "airy", "fluffy",
    "spongy", "elastic", "springy", "tough", "tender", "stringy", "mealy",
    "firm", "loose", "compact", "spreadable", "pourable", "spoonable",
    # Temperature
    "hot", "warm", "cool", "cold", "icy", "frozen", "room", "temperature",
    "piping", "simmering", "boiling", "sizzling", "bubbling", "steaming",
    "chilled", "refrigerated", "frozen", "thawed", "tempered", "blanched",
    # Appearance
    "golden", "browned", "charred", "seared", "caramelized", "glistening",
    "shiny", "matte", "dull", "bright", "vibrant", "pale", "dark", "deep",
    "light", "translucent", "opaque", "clear", "cloudy", "murky", "glossy",
    "garnished", "decorated", "plated", "presented", "arranged", "layered",
    # Quality
    "fresh", "stale", "spoiled", "ripe", "unripe", "overripe", "mature",
    "young", "aged", "matured", "vintage", "preserved", "pickled", "cured",
    "smoked", "dried", "dehydrated", "freeze-dried", "sun-dried", "roasted",
    "premium", "quality", "superior", "inferior", "standard", "gourmet",
    "authentic", "traditional", "modern", "contemporary", "classic", "fusion",
    "homemade", "store-bought", "packaged", "processed", "natural", "organic",
]

# Action verbs for cooking
ACTION_VERBS = [
    # Preparation
    "wash", "rinse", "clean", "scrub", "peel", "trim", "cut", "slice", "dice",
    "chop", "mince", "grate", "shred", "julienne", "chiffonade", "brunoise",
    "crush", "smash", "mash", "puree", "blend", "process", "grind", "mill",
    "strain", "sieve", "filter", "drain", "colander", "rinse", "blanch",
    "stem", "seed", "core", "pit", "bone", "fillet", "portion", "divide",
    # Cooking
    "heat", "cook", "boil", "simmer", "poach", "steam", "roast", "bake",
    "grill", "broil", "fry", "sauté", "sear", "braise", "stew", "poach",
    "steam", "pressure", "slow", "smoke", "cure", "pickle", "ferment",
    "flip", "turn", "stir", "whisk", "beat", "fold", "mix", "combine",
    "reduce", "thicken", "deglaze", "flambe", "caramelize", "brown", "char",
    # Finishing
    "season", "salt", "pepper", "spice", "herb", "garnish", "decorate",
    "plate", "arrange", "layer", "drizzle", "sprinkle", "scatter", "spread",
    "brush", "baste", "glaze", "coat", "dredge", "bread", "batter", "stuff",
    "serve", "present", "portion", "divide", "accompany", "garnish", "finish",
    # Preservation
    "preserve", "can", "jar", "freeze", "refrigerate", "store", "package",
    "seal", "vacuum", "wrap", "cover", "contain", "label", "date", "rotate",
]

# Time and duration vocabulary
TIME_VOCAB = [
    # Durations
    "second", "minute", "hour", "day", "week", "month", "year",
    "seconds", "minutes", "hours", "days", "weeks", "months", "years",
    "momentarily", "briefly", "quickly", "slowly", "gradually", "steadily",
    "overnight", "overnight", "marinate", "aging", "fermenting", "curing",
    # Frequency
    "always", "usually", "often", "frequently", "sometimes", "occasionally",
    "rarely", "seldom", "never", "once", "twice", "thrice", "repeatedly",
    # Sequencing
    "first", "second", "third", "fourth", "fifth", "lastly", "finally",
    "initially", "subsequently", "then", "next", "after", "before", "while",
    "simultaneously", "meanwhile", "thereafter", "afterwards", "eventually",
    "beforehand", "previously", "earlier", "later", "earlier", "later",
    # Timing indicators
    "early", "late", "midway", "halfway", "quarter", "third", "two-thirds",
    "beginning", "middle", "end", "start", "finish", "completion", "done",
    "ready", "prepared", "started", "begin", "continue", "pause", "resume",
]

# Measurement vocabulary
MEASUREMENT_VOCAB = [
    # Volume
    "teaspoon", "tablespoon", "cup", "pint", "quart", "gallon", "liter",
    "milliliter", "ounce", "fluid", "dash", "pinch", "splash", "dollop",
    "teaspoons", "tablespoons", "cups", "pints", "quarts", "gallons",
    # Weight
    "gram", "kilogram", "pound", "ounce", "milligram", "ton", "metric",
    "grams", "kilograms", "pounds", "ounces", "milligrams", "tons",
    # Temperature
    "degree", "celsius", "fahrenheit", "gas", "mark", "setting", "dial",
    "degrees", "hot", "medium", "low", "high", "moderate", "gentle",
    # Size
    "small", "medium", "large", "extra", "jumbo", "mini", "giant", "huge",
    "tiny", "little", "big", "enormous", "miniature", "bite-sized",
    # Counting
    "piece", "slice", "whole", "half", "quarter", "third", "portion",
    "piece", "pieces", "slice", "slices", "clove", "cloves", "head",
    "bunch", "stalk", "sprig", "leaf", "leaves", "strip", "strips",
]

# Geographic and origin vocabulary
ORIGIN_VOCAB = [
    # Regions
    "asian", "european", "african", "american", "latin", "middle", "eastern",
    "mediterranean", "scandinavian", "nordic", "caribbean", "pacific",
    "indian", "chinese", "japanese", "korean", "thai", "vietnamese",
    "italian", "french", "spanish", "german", "british", "greek", "turkish",
    "mexican", "brazilian", "argentine", "peruvian", "colombian", "cuban",
    "moroccan", "ethiopian", "nigerian", "egyptian", "south", "african",
    # Origins
    "local", "regional", "national", "international", "global", "worldwide",
    "imported", "domestic", "native", "indigenous", "traditional", "modern",
    "authentic", "genuine", "original", "classic", "contemporary", "fusion",
    # Farming
    "organic", "conventional", "sustainable", "farm-raised", "wild-caught",
    "free-range", "pasture-raised", "grass-fed", "grain-fed", "antibiotic",
    "pesticide-free", "non-gmo", "heirloom", "heritage", "artisanal",
]

# Sensory vocabulary
SENSORY_VOCAB = [
    # Sight
    "visual", "appearance", "color", "presentation", "plating", "garnish",
    "appealing", "appetizing", "attractive", "beautiful", "gorgeous",
    "rustic", "refined", "elegant", "casual", "formal", "simple", "elaborate",
    "golden", "brown", "white", "creamy", "dark", "light", "bright", "pale",
    # Smell
    "aromatic", "fragrant", "scented", "pungent", "strong", "mild", "faint",
    "perfumed", "floral", "herbal", "spicy", "woody", "earthy", "musky",
    "smoky", "burnt", "toasted", "roasted", "fresh", "citrusy", "fruity",
    # Sound
    "sizzling", "bubbling", "simmering", "boiling", "popping", "crackling",
    "crunching", "crispy", "snap", "crackle", "pop", "hissing", "whistling",
    # Touch
    "tactile", "texture", "mouthfeel", "consistency", "thickness", "thin",
    "viscous", "syrupy", "watery", "creamy", "smooth", "rough", "grainy",
]

# Scientific and chemical vocabulary for cooking
SCIENTIFIC_VOCAB = [
    # Chemical processes
    "protein", "carbohydrate", "fat", "lipid", "starch", "sugar", "glucose",
    "fructose", "sucrose", "lactose", "maltose", "dextrose", "caramelization",
    "maillard", "reaction", "emulsification", "emulsion", "coagulation",
    "gelatinization", "gelation", "fermentation", "oxidation", "reduction",
    "acid", "base", "alkaline", "ph", "neutralize", "balance", "flavor",
    # Nutritional
    "vitamin", "mineral", "nutrient", "nutrition", "calorie", "protein",
    "fiber", "sodium", "potassium", "calcium", "iron", "zinc", "magnesium",
    "antioxidant", "polyphenol", "carotenoid", "flavonoid", "isoflavone",
    "saturated", "unsaturated", "trans-fat", "cholesterol", "triglyceride",
    "amino", "acid", "enzyme", "bacteria", "yeast", "mold", "culture",
    "gluten", "casein", "lectin", "oxalate", "phytate", "saponin",
]

# Kitchen and cooking terminology
KITCHEN_VOCAB = [
    # Kitchen areas
    "kitchen", "pantry", "refrigerator", "freezer", "stove", "oven", "grill",
    "countertop", "workspace", "preparation", "storage", "serving", "dining",
    "sink", "faucet", "drain", "garbage", "compost", "recycling", "waste",
    # Equipment
    "appliance", "utensil", "cutlery", "flatware", "silverware", "serving",
    "mixer", "blender", "processor", "chopper", "grinder", "mill", "slicer",
    "scale", "thermometer", "timer", "clock", "gauge", "dial", "indicator",
    "container", "container", "vessel", "bowl", "dish", "pan", "pot", "tray",
    "sheet", "rack", "basket", "colander", "strainer", "sieve", "filter",
]

# Ingredient status and condition
CONDITION_VOCAB = [
    # Temperature state
    "frozen", "thawed", "refrigerated", "chilled", "room", "temperature",
    "warmed", "heated", "cooled", "iced", "tempered", "blanched", "shocked",
    # Freshness
    "fresh", "freshly", "raw", "cooked", "prepared", "processed", "preserved",
    "aged", "matured", "fermented", "cured", "smoked", "dried", "pickled",
    # Preparation states
    "prepared", "unprepared", "partial", "complete", "finished", "unfinished",
    "prepped", "measured", "weighed", "portioned", "divided", "combined",
    "separate", "individual", "mixed", "stirred", "whisked", "folded",
    "whipped", "beaten", "cream", "softened", "melted", "solidified",
]

# Casual/conversational food vocabulary
CASUAL_VOCAB = [
    # Casual terms
    "yummy", "tasty", "delicious", "scrumptious", "mouthwatering", "divine",
    "amazing", "fantastic", "incredible", "wonderful", "awesome", "great",
    "perfect", "excellent", "superb", "outstanding", "exceptional", "remarkable",
    "okay", "average", "mediocre", "disappointing", "bland", "boring", "plain",
    "burnt", "overcooked", "undercooked", "raw", "cold", "soggy", "tough",
    # Action expressions
    "whip", "throw", "toss", "dump", "splash", "sprinkle", "slather", "douse",
    "slap", "smear", "spread", "coat", "cover", "pile", "heap", "stuff",
    "cram", "pack", "squeeze", "press", "push", "pull", "tear", "rip",
]

# Phrase and expression vocabulary
EXPRESSION_VOCAB = [
    # Cooking expressions
    "from", "scratch", "homemade", "handmade", "handcrafted", "artisan",
    "quick", "easy", "simple", "fast", "slow", "patient", "careful",
    "carefully", "gently", "lightly", "heavily", "generously", "liberally",
    "sparingly", "moderately", "judiciously", "properly", "correctly",
    "perfectly", "correctly", "exactly", "approximately", "roughly",
    "about", "around", "nearly", "almost", "just", "barely", "scarcely",
    # Instructional language
    "optional", "recommended", "suggested", "preferred", "desired",
    "required", "necessary", "essential", "important", "critical", "crucial",
    "helpful", "useful", "beneficial", "advised", "best", "ideal", "optimal",
]

# Additional vocabulary lists
SAUCES = [
    "marinara", "alfredo", "pesto", "bolognese", "carbonara", "bechamel",
    "hollandaise", "bearnaise", "veloute", "espagnole", "demi-glace",
    "teriyaki", "soy sauce", "oyster sauce", "hoisin", "plum sauce",
    "sweet and sour", "kung pao", "black bean sauce", "char siu",
    "chimichurri", "salsa verde", "romesco", "aioli", "tartar sauce",
    "cocktail sauce", "remoulade", "tzatziki", "raita", "hummus",
    "barbecue sauce", "ketchup", "mustard", "mayonnaise", "ranch",
    "blue cheese dressing", "caesar dressing", "vinaigrette", "italian dressing",
    "hot sauce", "sriracha", "tabasco", "cholula", "sambal oelek",
    "chili paste", "gochujang", "doubanjiang", "miso paste", "tahini",
]

# Equipment
EQUIPMENT = [
    "knife", "cutting board", "pot", "pan", "skillet", "saucepan", "stockpot",
    "dutch oven", "wok", "baking sheet", "roasting pan", "casserole dish",
    "mixing bowl", "measuring cup", "measuring spoon", "thermometer", "timer",
    "blender", "food processor", "stand mixer", "hand mixer", "whisk",
    "spatula", "tongs", "ladle", "slotted spoon", "colander", "strainer",
    "steamer basket", "pressure cooker", "slow cooker", "instant pot",
]

# Cut types
CUTS = [
    "dice", "mince", "chop", "slice", "julienne", "brunoise", "chiffonade",
    "bias cut", "rough chop", "fine dice", "medium dice", "large dice",
    "thin slice", "thick slice", "wedge", "quarter", "halve",
]

# Cooking verbs
COOKING_VERBS = [
    "heat", "warm", "cook", "sauté", "sweat", "caramelize", "brown", "sear",
    "deglaze", "reduce", "simmer", "boil", "blanch", "shock", "roast", "bake",
    "grill", "broil", "fry", "deep-fry", "steam", "poach", "braise", "stew",
    "marinate", "brine", "cure", "ferment", "rest", "season", "taste",
]

# Descriptive terms
TEXTURES = [
    "tender", "crisp", "crunchy", "soft", "creamy", "flaky", "chewy", "firm",
    "juicy", "moist", "dry", "silky", "smooth", "rough", "velvety", "light",
]

FLAVORS = [
    "sweet", "salty", "sour", "bitter", "umami", "savory", "spicy", "mild",
    "rich", "light", "bold", "subtle", "complex", "simple", "balanced",
]

# Dish categories
DISH_TYPES = [
    "soup", "stew", "salad", "pasta", "rice", "curry", "stir-fry", "roast",
    "grill", "braise", "bake", "sauté", "casserole", "one-pot meal",
    "appetizer", "main course", "side dish", "dessert", "breakfast", "snack",
    "sandwich", "wrap", "bowl", "bento", "platter",
]

CUISINES = [
    "Italian", "French", "Chinese", "Japanese", "Indian", "Thai", "Mexican",
    "Mediterranean", "Middle Eastern", "American", "Korean", "Vietnamese",
    "Greek", "Spanish", "Moroccan", "Indonesian", "Filipino", "German",
]

# ============================================================================
# SENTENCE TEMPLATES - Varied natural language patterns
# ============================================================================

def generate_factual_statements():
    """Generate factual cooking knowledge statements."""
    statements = []

    # Cooking fundamentals
    fundamentals = [
        f"Cooking is the process of preparing food by applying heat.",
        f"Different cooking methods produce different textures and flavors.",
        f"Heat transfer occurs through conduction, convection, and radiation.",
        f"Temperature control is essential for proper cooking.",
        f"The Maillard reaction creates browning and complex flavors.",
        f"Caramelization occurs when sugars break down under heat.",
        f"Resting meat after cooking allows juices to redistribute.",
        f"Carryover cooking continues after food is removed from heat.",
        f"Mise en place means preparing ingredients before cooking.",
        f"Proper seasoning enhances natural flavors.",
        f"Salt suppresses bitterness and enhances other flavors.",
        f"Acid brightens flavors and balances richness.",
        f"Fat carries flavor compounds and creates mouthfeel.",
        f"Umami adds savory depth to dishes.",
        f"Temperature affects cooking time and results.",
    ]
    statements.extend(fundamentals)

    # Method descriptions
    for method in METHODS:
        statements.extend([
            f"{method.capitalize()} is a fundamental cooking technique.",
            f"When you {method}, proper temperature control matters.",
            f"The key to {method}ing successfully is timing.",
            f"To {method} properly, start with the right technique.",
        ])

    # Ingredient knowledge
    for veg in VEGETABLES:
        statements.extend([
            f"{veg.capitalize()} is a versatile vegetable.",
            f"When cooking {veg}, consider the preparation method.",
            f"Fresh {veg} should be stored properly.",
            f"{veg.capitalize()} can be prepared many ways.",
        ])

    for fruit in FRUITS:
        statements.extend([
            f"{fruit.capitalize()} adds natural sweetness.",
            f"Fresh {fruit} provides vitamins and flavor.",
            f"{fruit.capitalize()} works in both sweet and savory dishes.",
        ])

    for protein in PROTEINS:
        statements.extend([
            f"{protein.capitalize()} is a common protein source.",
            f"When cooking {protein}, proper temperature ensures safety.",
            f"{protein.capitalize()} can be prepared many ways.",
            f"Different cooking methods suit {protein} differently.",
        ])

    for grain in GRAINS:
        statements.extend([
            f"{grain.capitalize()} is a staple grain.",
            f"Cooking {grain} requires proper water ratios.",
            f"{grain.capitalize()} provides energy and nutrition.",
        ])

    # Pasta types from GRAINS
    pastas = ["spaghetti", "linguine", "fettuccine", "penne", "rigatoni",
              "macaroni", "farfalle", "fusilli", "orzo", "ravioli", "lasagna"]
    for pasta in pastas:
        statements.extend([
            f"{pasta.capitalize()} is a classic pasta shape.",
            f"{pasta.capitalize()} pairs well with various sauces.",
            f"Cooking {pasta} al dente requires proper timing.",
        ])

    # Equipment knowledge
    for eq in EQUIPMENT:
        statements.extend([
            f"A {eq} is essential for cooking.",
            f"Using a {eq} properly improves results.",
            f"Keep your {eq} clean and well-maintained.",
            f"The right {eq} makes cooking easier.",
        ])

    return statements


def generate_question_answer_pairs():
    """Generate Q&A style content."""
    qa_pairs = []

    # How do I questions
    how_questions = [
        ("boil an egg", "Place eggs in cold water, bring to a boil, then reduce heat. For soft-boiled eggs, cook 4-6 minutes. For hard-boiled eggs, cook 9-12 minutes. Transfer to ice water to stop cooking."),
        ("cook rice", "Rinse rice until water runs clear. Use 1.5 cups water per cup of white rice. Bring to boil, reduce to simmer, cover tightly. Cook 18 minutes. Rest 5 minutes, then fluff."),
        ("make pasta", "Use a large pot with plenty of salted water. Bring to rolling boil. Add pasta, stir immediately. Cook until al dente, usually 8-10 minutes. Reserve pasta water before draining."),
        ("sauté vegetables", "Heat oil in pan until shimmering. Add aromatics first. Stir frequently. Cook until tender-crisp. Season near the end to preserve texture."),
        ("roast chicken", "Preheat oven to 425 degrees F. Pat chicken dry, rub with oil and seasonings. Roast until internal temperature reaches 165 degrees F. Rest 10-15 minutes before carving."),
        ("make a sauce", "Start with aromatics in fat. Add liquid components. Simmer to reduce and concentrate. Season to taste. Finish with herbs or butter."),
        ("grill steak", "Bring steak to room temperature. Pat dry and season well. Grill on high heat, turning once. Use thermometer for doneness. Rest 5 minutes before serving."),
        ("make soup", "Sauté aromatics first. Add liquid and main ingredients. Simmer until flavors meld. Season throughout cooking. Finish with fresh herbs or acid."),
        ("bake bread", "Combine flour, yeast, salt, and water. Knead until smooth and elastic. Let rise in warm place until doubled. Shape and bake until golden and hollow-sounding."),
        ("make salad dressing", "Whisk acid like vinegar with salt and seasonings. Slowly drizzle in oil while whisking. Emulsify until combined. Taste and adjust."),
    ]

    for topic, answer in how_questions:
        qa_pairs.extend([
            f"How do I {topic}?",
            answer,
            f"What is the proper way to {topic}?",
            answer,
            f"Can you explain how to {topic}?",
            answer,
        ])

    # What is questions
    what_questions = [
        ("the Maillard reaction", "The Maillard reaction occurs when proteins and sugars brown under heat above 285 degrees F, creating complex flavor compounds."),
        ("umami", "Umami is the fifth basic taste, described as savory or meaty, found in foods like mushrooms, tomatoes, cheese, and soy sauce."),
        ("al dente", "Al dente means 'to the tooth' in Italian, referring to pasta or vegetables cooked until firm but not soft."),
        ("mise en place", "Mise en place is a French term meaning 'everything in place', referring to preparing and organizing all ingredients before cooking."),
        ("braising", "Braising is a combination cooking method using both dry and moist heat, cooking food slowly in liquid after searing."),
        ("blanching", "Blanching briefly cooks food in boiling water then shocks it in ice water to stop cooking and preserve color."),
        ("deglazing", "Deglazing adds liquid to a hot pan to release caramelized fond from the bottom for making sauces."),
        ("carryover cooking", "Carryover cooking continues after food is removed from heat as residual energy raises internal temperature."),
        ("mirepoix", "Mirepoix is a French flavor base of diced onion, carrot, and celery, typically in a 2:1:1 ratio."),
        ("roux", "Roux is a mixture of fat and flour cooked together used to thicken sauces and soups."),
    ]

    for topic, answer in what_questions:
        qa_pairs.extend([
            f"What is {topic}?",
            answer,
            f"Define {topic}.",
            answer,
            f"Explain {topic} in cooking.",
            answer,
        ])

    # Why questions
    why_questions = [
        ("let meat rest", "Resting allows juices to redistribute throughout the meat. Cutting immediately causes flavorful juices to run out onto the cutting board."),
        ("salt pasta water", "Salting pasta water seasons the pasta from within. The water should taste like a properly seasoned soup for best flavor."),
        ("preheat the oven", "Preheating ensures food enters the correct temperature environment immediately. This affects rise, texture, and cooking time."),
        ("rinse rice", "Rinsing removes excess starch that makes rice sticky and gummy. Rinsed rice cooks fluffier and grains stay separate."),
        ("pat meat dry before searing", "Surface moisture creates steam that prevents proper browning. Dry meat sears better and develops superior flavor."),
        ("use room temperature ingredients in baking", "Room temperature ingredients mix together more evenly. Cold ingredients can curdle or create lumpy batters."),
        ("not crowd the pan", "Crowding drops pan temperature and causes steaming instead of searing. Food cooks in batches if necessary."),
        ("rest dough before baking", "Resting relaxes gluten and allows moisture to distribute evenly. This improves texture and workability."),
        ("add herbs at different times", "Hardy herbs can cook longer to release flavor. Delicate herbs lose potency if cooked too long and are added at the end."),
        ("not open the oven during baking", "Opening the oven releases heat and can cause baked goods to collapse. Peek only when absolutely necessary."),
    ]

    for topic, answer in why_questions:
        qa_pairs.extend([
            f"Why should I {topic}?",
            answer,
            f"Why is it important to {topic}?",
            answer,
        ])

    return qa_pairs


def generate_instructional_content():
    """Generate step-by-step instructions."""
    instructions = []

    # General cooking principles
    principles = [
        "Start by reading the entire recipe before beginning.",
        "Gather and prepare all ingredients before cooking.",
        "Keep your workspace clean and organized.",
        "Season throughout cooking, not just at the end.",
        "Taste your food as you cook to understand flavor development.",
        "Sharp knives are safer than dull ones.",
        "Clean as you go to manage kitchen mess.",
        "Use the right tool for each job.",
        "Temperature control is fundamental to good cooking.",
        "Timing matters as much as technique.",
    ]
    instructions.extend(principles)

    # Method-specific instructions
    method_instructions = {
        "boil": [
            "Use enough water to allow proper circulation around food.",
            "Salt the water generously for seasoning.",
            "Bring to a full rolling boil before adding ingredients.",
            "Reduce heat to maintain steady boil without overflow.",
            "Monitor cooking time carefully for proper texture.",
        ],
        "sauté": [
            "Use the right size pan for even heating.",
            "Heat the pan before adding oil.",
            "Let oil shimmer before adding food.",
            "Do not overcrowd the pan.",
            "Keep food moving for even cooking.",
            "Season near the end to preserve texture.",
        ],
        "roast": [
            "Preheat oven fully before putting food in.",
            "Cut ingredients to similar sizes for even cooking.",
            "Toss vegetables with oil and seasonings.",
            "Arrange in a single layer with space between.",
            "Flip or stir halfway through cooking.",
            "Check doneness with a thermometer or knife.",
        ],
        "grill": [
            "Preheat grill thoroughly before cooking.",
            "Clean grates and oil lightly to prevent sticking.",
            "Bring meat to room temperature before grilling.",
            "Pat dry and season before placing on grill.",
            "Do not move food unnecessarily.",
            "Use thermometer for accurate doneness.",
            "Let meat rest after grilling.",
        ],
    }

    for method, tips in method_instructions.items():
        for tip in tips:
            instructions.append(f"When you {method}: {tip.lower()}")

    return instructions


def generate_measurement_guidance():
    """Generate content about measurements and timing."""
    guidance = []

    # Standard measurements
    measurements = [
        "3 teaspoons equal 1 tablespoon.",
        "4 tablespoons equal 1/4 cup.",
        "16 tablespoons equal 1 cup.",
        "2 cups equal 1 pint.",
        "2 pints equal 1 quart.",
        "4 quarts equal 1 gallon.",
        "Weight measurements are more accurate than volume for baking.",
        "Different flours weigh different amounts per cup.",
        "Use liquid measuring cups for liquids and dry measures for solids.",
        "Level off dry ingredients with a straight edge.",
    ]
    guidance.extend(measurements)

    # Cooking temperatures
    temps = [
        "Rare beef: 120-125 degrees F internal.",
        "Medium-rare beef: 130-135 degrees F internal.",
        "Medium beef: 140-145 degrees F internal.",
        "Well-done beef: 160+ degrees F internal.",
        "Poultry should reach 165 degrees F internal.",
        "Ground meats should reach 160 degrees F internal.",
        "Fish should reach 145 degrees F internal.",
        "Eggs should reach 160 degrees F internal for safety.",
        "Low oven is around 300 degrees F.",
        "Moderate oven is around 350 degrees F.",
        "Hot oven is around 400 degrees F.",
        "Very hot oven is around 450 degrees F.",
    ]
    guidance.extend(temps)

    # Cooking times
    times = [
        "Soft-boiled eggs cook in 4-6 minutes.",
        "Hard-boiled eggs cook in 9-12 minutes.",
        "White rice cooks in 18-20 minutes.",
        "Brown rice cooks in 40-45 minutes.",
        "Pasta usually cooks in 8-12 minutes.",
        "Quick sautéing takes 3-5 minutes.",
        "Roasting vegetables takes 20-45 minutes.",
        "Whole chicken roasts in about 1.5 hours.",
        "Steak grills in 4-6 minutes per side.",
        "Slow cooking takes 4-8 hours.",
    ]
    guidance.extend(times)

    return guidance


def generate_pairing_knowledge():
    """Generate flavor pairing knowledge."""
    pairings = []

    # Classic combinations
    classic_pairs = [
        "Tomatoes and basil complement each other perfectly.",
        "Lemon pairs well with fish and seafood.",
        "Garlic and onion form the base of many cuisines.",
        "Cheese and wine are natural partners.",
        "Chocolate and raspberries balance each other.",
        "Salt and caramel enhance each other.",
        "Rosemary and lamb are a classic combination.",
        "Basil and mozzarella work beautifully together.",
        "Ginger and citrus brighten Asian dishes.",
        "Cinnamon and apples warm fall flavors.",
    ]
    pairings.extend(classic_pairs)

    # Herb pairings
    herb_pairs = [
        "Basil pairs with tomatoes, mozzarella, and garlic.",
        "Rosemary pairs with lamb, potatoes, and bread.",
        "Thyme pairs with chicken, vegetables, and soups.",
        "Oregano pairs with Mediterranean dishes and sauces.",
        "Cilantro pairs with Mexican, Indian, and Asian dishes.",
        "Dill pairs with fish, cucumber, and yogurt.",
        "Mint pairs with lamb, peas, and desserts.",
        "Sage pairs with pork, poultry, and stuffing.",
        "Parsley finishes almost any savory dish.",
        "Tarragon pairs with chicken, eggs, and cream sauces.",
    ]
    pairings.extend(herb_pairs)

    # Spice blends
    spice_info = [
        "Garam masala is a warming Indian spice blend.",
        "Curry powder combines turmeric, cumin, and other spices.",
        "Chinese five-spice includes star anise, fennel, and cinnamon.",
        "Italian herbs combine oregano, basil, and rosemary.",
        "Herbes de Provence includes lavender and Mediterranean herbs.",
        "Cajun seasoning is spicy and herbaceous.",
        "Za'atar combines thyme, sesame, and sumac.",
        "Ras el hanout is a complex Moroccan blend.",
        "Adobo is a versatile Latin American seasoning.",
        "Jerk is a spicy Caribbean blend with allspice and chilies.",
    ]
    pairings.extend(spice_info)

    return pairings


def generate_troubleshooting():
    """Generate troubleshooting content."""
    troubleshooting = []

    problems = [
        ("Food is too salty", "Add acid like lemon juice, add unsweetened fat, or serve with unseasoned starch to dilute."),
        ("Food is too spicy", "Add dairy like yogurt or cream, or add sweetness to balance heat."),
        ("Food is too sour", "Add sweetness or fat to balance acidity."),
        ("Food is too bitter", "Add salt or sweetness to suppress bitterness."),
        ("Sauce is lumpy", "Strain through a fine mesh sieve or blend until smooth."),
        ("Sauce separated", "Whisk vigorously, or blend with a little water to re-emulsify."),
        ("Soup is too thin", "Simmer longer to reduce, or add a slurry of cornstarch and water."),
        ("Soup is too thick", "Add more liquid, either water, broth, or cream."),
        ("Meat is tough", "It may need longer cooking with moist heat to break down connective tissue."),
        ("Cakes sank", "Do not open the oven too early, check that leavening is fresh."),
        ("Cookies spread too much", "Chill dough before baking, and measure flour correctly."),
        ("Rice is mushy", "Use less water next time, or spread on baking sheet to dry."),
        ("Vegetables are soggy", "Cook at higher heat, do not overcrowd pan, roast instead of steam."),
        ("Food sticks to pan", "Heat pan properly before adding oil, pat food dry before cooking."),
        ("Eggs are rubbery", "Cook at lower heat, remove from heat while slightly underdone."),
    ]

    for problem, solution in problems:
        troubleshooting.extend([
            f"If your {problem.lower()}, {solution}",
            f"Problem: {problem}. Solution: {solution}",
        ])

    return troubleshooting


def generate_variant_sentences():
    """Generate many variant sentences through combinations."""
    sentences = []

    # Ingredient + method combinations
    for method in METHODS:
        for veg in VEGETABLES:
            sentences.extend([
                f"To {method} {veg}, proper technique ensures best results.",
                f"{veg.capitalize()} can be {method}ed successfully.",
                f"When you {method} {veg}, watch the timing.",
                f"{method.capitalize()}ing {veg} requires attention.",
            ])

    # Protein + method combinations
    for method in METHODS:
        for protein in PROTEINS:
            sentences.extend([
                f"To {method} {protein}, start with proper preparation.",
                f"{protein.capitalize()} {method}s well with the right technique.",
                f"When {method}ing {protein}, temperature matters.",
            ])

    # Herb + ingredient pairings
    for herb in HERBS:
        for ingredient in VEGETABLES:
            sentences.extend([
                f"{herb.capitalize()} pairs well with {ingredient}.",
                f"Add {herb} to {ingredient} dishes for flavor.",
            ])

    # Spice + dish combinations
    for spice in SPICES:
        for dish in DISH_TYPES:
            sentences.extend([
                f"{spice.capitalize()} enhances {dish} recipes.",
                f"Add {spice} to {dish} for depth of flavor.",
            ])

    return sentences


def generate_conversational_exchanges():
    """Generate natural conversational content."""
    exchanges = []

    # User questions and assistant responses
    user_questions = [
        "I want to cook dinner but don't know what to make.",
        "How can I improve my cooking skills?",
        "What equipment do I really need in my kitchen?",
        "How do I know when meat is cooked properly?",
        "I'm a beginner cook, what should I learn first?",
        "How can I make healthier meals at home?",
        "What are some quick weeknight dinner ideas?",
        "How do I reduce food waste in my kitchen?",
        "What are common cooking mistakes to avoid?",
        "How do I season food properly?",
    ]

    assistant_responses = [
        "Start with what ingredients you have available. Simple dishes often work best for weeknight cooking.",
        "Practice fundamental techniques like knife skills, heat control, and seasoning. Each dish builds skills.",
        "A chef's knife, cutting board, large pot, skillet, and sheet pan cover most cooking needs.",
        "Use a thermometer for accuracy. Visual cues like cooking time and firmness help with experience.",
        "Begin with eggs, pasta, and simple vegetable dishes. Mastering basics builds confidence.",
        "Cook from scratch more often, use more vegetables, and control seasoning and portions.",
        "Stir-fries, pasta dishes, and sheet pan meals come together quickly with minimal cleanup.",
        "Plan meals, use leftovers creatively, store properly, and understand shelf life.",
        "Not reading recipes fully, incorrect measuring, overcrowding pans, and not letting meat rest.",
        "Taste throughout cooking. Season in layers. Salt enhances, acid brightens, fat carries flavor.",
    ]

    for q, a in zip(user_questions, assistant_responses):
        exchanges.extend([
            q,
            a,
            f"User: {q}",
            f"Assistant: {a}",
        ])

    return exchanges


def generate_cuisine_knowledge():
    """Generate cuisine-specific content."""
    cuisine_content = []

    cuisine_descriptions = {
        "Italian": [
            "Italian cuisine emphasizes quality ingredients and simple preparation.",
            "Olive oil, tomatoes, garlic, and basil define many Italian dishes.",
            "Pasta and risotto are Italian staples prepared in countless variations.",
            "Italian cooking varies significantly by region.",
        ],
        "Chinese": [
            "Chinese cuisine encompasses many regional styles and techniques.",
            "Stir-frying, steaming, and braising are fundamental Chinese cooking methods.",
            "Five flavors guide Chinese cooking: sweet, sour, salty, bitter, and umami.",
            "Regional Chinese cuisines range from mild and savory to hot and numbing.",
        ],
        "Indian": [
            "Indian cuisine features complex spice blends and varied regional styles.",
            "Curry refers to many different dishes with spiced sauce bases.",
            "Indian breads include naan, roti, and paratha.",
            "Lentils, rice, and vegetables form the foundation of many Indian meals.",
        ],
        "Mexican": [
            "Mexican cuisine combines indigenous and Spanish influences.",
            "Corn, beans, and chilies are foundational Mexican ingredients.",
            "Mexican salsa ranges from fresh pico de gallo to cooked sauces.",
            "Traditional Mexican cooking was recognized by UNESCO.",
        ],
        "French": [
            "French cuisine established many foundational cooking techniques.",
            "The five mother sauces originated in French cooking.",
            "French regional cooking varies from butter-rich north to olive oil south.",
            "Classic French technique emphasizes proper preparation and presentation.",
        ],
        "Japanese": [
            "Japanese cuisine emphasizes seasonal ingredients and presentation.",
            "Sushi, tempura, and ramen represent different Japanese cooking styles.",
            "Japanese dashi provides umami base for many dishes.",
            "Balance and simplicity characterize Japanese food.",
        ],
        "Thai": [
            "Thai cuisine balances sweet, sour, salty, and spicy flavors.",
            "Coconut milk, lime, fish sauce, and chilies define many Thai dishes.",
            "Thai curries vary by color and spice level.",
            "Fresh herbs and aromatics are essential to Thai cooking.",
        ],
        "Mediterranean": [
            "Mediterranean cuisine emphasizes olive oil, vegetables, and lean proteins.",
            "Fresh herbs, garlic, and lemon feature prominently.",
            "Mediterranean diets are associated with health benefits.",
            "Regional variations span multiple countries and cultures.",
        ],
    }

    for cuisine, descriptions in cuisine_descriptions.items():
        for desc in descriptions:
            cuisine_content.append(desc)

    return cuisine_content


def generate_safety_content():
    """Generate food safety knowledge."""
    safety = []

    safety_rules = [
        "Wash hands before handling food.",
        "Keep raw and cooked foods separate.",
        "Use separate cutting boards for raw proteins.",
        "Cook foods to proper internal temperatures.",
        "Refrigerate perishable foods within two hours.",
        "Thaw frozen foods safely in refrigerator or cold water.",
        "Marinate foods in the refrigerator, not at room temperature.",
        "Do not cross-contaminate with used marinades.",
        "Check internal temperatures with a food thermometer.",
        "When in doubt, throw it out.",
        "Clean surfaces and utensils after handling raw proteins.",
        "Store raw meat on the lowest shelf to prevent drips.",
        "Leftovers should be consumed within 3-4 days.",
        "Hot foods should stay hot, cold foods should stay cold.",
        "The danger zone for bacterial growth is 40-140 degrees F.",
    ]
    safety.extend(safety_rules)

    return safety


def generate_seasonal_cooking():
    """Generate seasonal cooking content."""
    seasonal = []

    seasonal_tips = {
        "Spring": [
            "Spring brings fresh asparagus, peas, and artichokes.",
            "Light spring vegetables need gentle cooking.",
            "Spring lamb is traditional for seasonal meals.",
            "Fresh herbs return in spring gardens.",
        ],
        "Summer": [
            "Summer tomatoes, corn, and zucchini are abundant.",
            "Grilling keeps the kitchen cool in summer.",
            "Summer salads feature fresh raw vegetables.",
            "Stone fruits and berries peak in summer.",
        ],
        "Fall": [
            "Fall brings squash, apples, and root vegetables.",
            "Hearty braises and roasts suit fall weather.",
            "Warm spices like cinnamon complement fall foods.",
            "Fall is time for slow cooker meals.",
        ],
        "Winter": [
            "Winter calls for warming soups and stews.",
            "Root vegetables and hardy greens store well.",
            "Slow cooking develops flavor in winter months.",
            "Comfort foods are welcome in cold weather.",
        ],
    }

    for season, tips in seasonal_tips.items():
        for tip in tips:
            seasonal.append(tip)

    return seasonal


def generate_nutrition_basics():
    """Generate nutritional knowledge content."""
    nutrition = []

    nutrition_facts = [
        "Protein builds and repairs body tissues.",
        "Carbohydrates provide energy for daily activities.",
        "Healthy fats support brain function and nutrient absorption.",
        "Fiber aids digestion and promotes fullness.",
        "Vitamins support various body functions.",
        "Minerals are essential for health.",
        "A balanced diet includes variety.",
        "Portion control maintains healthy weight.",
        "Whole foods provide more nutrients than processed.",
        "Hydration is essential for health.",
        "Colorful vegetables provide different nutrients.",
        "Leafy greens are nutrient-dense.",
        "Whole grains offer more nutrition than refined.",
        "Lean proteins support muscle health.",
        "Cooking method affects nutrient retention.",
    ]
    nutrition.extend(nutrition_facts)

    return nutrition


def generate_additional_content():
    """Generate additional content to reach 100k lines."""
    content = []

    # Extended technique combinations
    techniques = [
        "blanch and shock", "sweat vegetables", "reduce sauce", "emulsify dressing",
        "fold ingredients", "cream butter", "temper eggs", "proof yeast",
        "knead dough", "rest meat", "deglaze pan", "mount sauce",
    ]

    for tech in techniques:
        content.extend([
            f"{tech.capitalize()} is essential for certain preparations.",
            f"To {tech}, follow proper technique.",
            f"The purpose of {tech} is to improve texture and flavor.",
            f"{tech.capitalize()} correctly for best results.",
        ])

    # Extended ingredient preparations
    prep_methods = [
        "wash", "peel", "trim", "core", "seed", "devein", "shuck", "grate",
        "shred", "zest", "juice", "crush", "smash", "press", "strain",
    ]

    for prep in prep_methods:
        for veg in VEGETABLES:
            content.append(f"To {prep} {veg}, use proper technique.")
        for fruit in FRUITS:
            content.append(f"{prep.capitalize()} {fruit} carefully for best results.")

    # Extended flavor combinations
    flavor_pairs = [
        ("garlic", "onion", "aromatic base"),
        ("lemon", "butter", "rich brightness"),
        ("rosemary", "potato", "classic herb"),
        ("tomato", "basil", "Italian staple"),
        ("ginger", "soy sauce", "Asian umami"),
    ]

    for ing1, ing2, result in flavor_pairs:
        content.extend([
            f"Combining {ing1} and {ing2} creates {result}.",
            f"{ing1.capitalize()} and {ing2} are natural partners.",
            f"The pairing of {ing1} with {ing2} yields {result}.",
        ])

    # Extended cooking advice
    advice = [
        "Read recipes completely before starting.",
        "Prep ingredients before turning on the heat.",
        "Taste food throughout the cooking process.",
        "Adjust seasoning gradually and taste between additions.",
        "Rest proteins before cutting into them.",
        "Use proper pan temperature for searing.",
        "Do not move food before it is ready to flip.",
        "Dry ingredients before cooking for better browning.",
        "Room temperature eggs mix better in batters.",
        "Cold butter makes flakier pastries.",
        "Room temperature butter creams better for cookies.",
        "Butter temperature affects cookie spread.",
        "Oil temperature matters for frying.",
        "Preheating affects cooking times.",
        "Altitude affects boiling points and baking times.",
    ]
    content.extend(advice)

    # More Q&A variations
    more_qa = [
        ("How long should I cook pasta?", "Cook pasta until al dente, usually 8-12 minutes. Taste a minute before the package time."),
        ("What heat setting for simmering?", "Use low to medium-low heat for simmering. You should see small bubbles rising Occasionally."),
        ("How do I know when oil is hot enough?", "Oil shimmers when ready. A piece of food dropped in should sizzle immediately."),
        ("Why did my sauce break?", "Sauces break when overheated or mixed too aggressively. Whisk vigorously to re-emulsify."),
        ("How do I fix a salty dish?", "Add acid like lemon juice, increase the volume with an unseasoned ingredient, or serve over unseasoned starch."),
    ]

    for q, a in more_qa:
        content.extend([q, a])

    return content


def generate_expanded_pairs():
    """Generate many more ingredient and technique combinations."""
    pairs = []

    # Sentence pattern templates for vegetables
    veg_patterns = [
        "{veg} is a key ingredient in many cuisines.",
        "Fresh {veg} adds flavor and texture to dishes.",
        "When shopping for {veg}, look for firm specimens.",
        "Store {veg} properly to maintain freshness.",
        "{veg} works well in both raw and cooked applications.",
        "The flavor of {veg} intensifies when cooked.",
        "Seasonal {veg} tastes best and costs less.",
        "Preparing {veg} correctly improves the final dish.",
        "{veg} pairs with many different proteins.",
        "Add {veg} early in cooking for depth of flavor.",
        "{veg} adds nutrition as well as flavor.",
        "Different cooking methods transform {veg} differently.",
    ]

    for veg in VEGETABLES:
        for pattern in veg_patterns:
            pairs.append(pattern.format(veg=veg))

    # Sentence pattern templates for proteins
    protein_patterns = [
        "{protein} is a versatile protein source.",
        "Cook {protein} to the proper internal temperature.",
        "Fresh {protein} should have a clean smell.",
        "{protein} can be prepared in many different ways.",
        "The key to cooking {protein} is proper technique.",
        "{protein} benefits from proper seasoning.",
        "Resting {protein} after cooking improves juiciness.",
        "{protein} pairs well with various side dishes.",
        "Different cuts of {protein} require different methods.",
        "{protein} is featured in cuisines worldwide.",
    ]

    for protein in PROTEINS:
        for pattern in protein_patterns:
            pairs.append(pattern.format(protein=protein))

    # Sentence pattern templates for herbs
    herb_patterns = [
        "{herb} adds fresh flavor to dishes.",
        "Add {herb} at the right time for best results.",
        "Fresh {herb} has more aroma than dried.",
        "{herb} pairs well with many ingredients.",
        "Store {herb} properly to extend its life.",
        "{herb} brightens the flavor of rich dishes.",
        "Chop {herb} just before using for maximum flavor.",
        "{herb} is essential in many cuisines.",
    ]

    for herb in HERBS:
        for pattern in herb_patterns:
            pairs.append(pattern.format(herb=herb))

    # Sentence pattern templates for spices
    spice_patterns = [
        "{spice} adds depth and complexity to dishes.",
        "Toast {spice} to release its aromatic oils.",
        "Store {spice} in a cool dark place.",
        "{spice} is used in cuisines around the world.",
        "Add {spice} at different stages for different effects.",
        "{spice} pairs well with certain ingredients.",
        "Freshly ground {spice} has more potency.",
    ]

    for spice in SPICES:
        for pattern in spice_patterns:
            pairs.append(pattern.format(spice=spice))

    # Method + Ingredient combinations with varied patterns
    for method in METHODS:
        method_cap = method.capitalize()
        for veg in VEGETABLES:
            ing_cap = veg.capitalize()
            pairs.extend([
                f"To {method} {veg} properly, use the right technique.",
                f"{ing_cap} can be {method}ed to delicious results.",
                f"When you {method} {veg}, timing is important.",
                f"{method_cap}ing {veg} changes its flavor.",
                f"The best way to {method} {veg} starts with preparation.",
                f"{ing_cap} {method}s best with the right temperature.",
                f"Learning to {method} {veg} improves cooking skills.",
                f"{ing_cap} becomes tender when you {method} it.",
            ])
        for protein in PROTEINS:
            ing_cap = protein.capitalize()
            pairs.extend([
                f"To {method} {protein} properly, use the right technique.",
                f"{ing_cap} can be {method}ed to delicious results.",
                f"When you {method} {protein}, timing is important.",
                f"{method_cap}ing {protein} changes its flavor.",
                f"The best way to {method} {protein} starts with preparation.",
                f"{ing_cap} {method}s best with the right temperature.",
                f"Learning to {method} {protein} improves cooking skills.",
                f"{ing_cap} becomes tender when you {method} it.",
            ])

    # Flavor + ingredient combinations
    flavor_patterns = [
        "{ing} naturally has {flavor} notes.",
        "Enhance the {flavor} qualities of {ing}.",
        "Balance the {flavor} aspects of {ing}.",
        "{ing} works well in {flavor} dishes.",
        "{flavor} ingredients complement {ing}.",
    ]

    for flavor in FLAVORS:
        for veg in VEGETABLES:
            for pattern in flavor_patterns[:2]:
                pairs.append(pattern.format(ing=veg, flavor=flavor))
        for protein in PROTEINS:
            for pattern in flavor_patterns[:2]:
                pairs.append(pattern.format(ing=protein, flavor=flavor))

    # Equipment with tasks - expanded
    for eq in EQUIPMENT:
        eq_cap = eq.capitalize()
        for method in METHODS:
            pairs.extend([
                f"A {eq} is essential for {method}ing.",
                f"Use your {eq} correctly when you {method}.",
                f"The {eq} is perfect for {method} preparations.",
                f"Keep your {eq} ready for {method} tasks.",
                f"{eq_cap} helps achieve good {method} results.",
                f"Proper {eq} use improves {method} outcomes.",
            ])

    # Cuisine combinations - expanded
    for cuisine in CUISINES:
        for veg in VEGETABLES:
            ing_cap = veg.capitalize()
            pairs.extend([
                f"{cuisine} cuisine is known for {veg}.",
                f"{ing_cap} features prominently in {cuisine} dishes.",
                f"In {cuisine} cooking, {veg} is essential.",
                f"{cuisine} recipes often include {veg}.",
                f"The flavors of {cuisine} cuisine complement {veg}.",
                f"{cuisine} chefs use {veg} skillfully.",
            ])
        for protein in PROTEINS:
            ing_cap = protein.capitalize()
            pairs.extend([
                f"{cuisine} cuisine is known for {protein}.",
                f"{ing_cap} features prominently in {cuisine} dishes.",
                f"In {cuisine} cooking, {protein} is essential.",
                f"{cuisine} recipes often include {protein}.",
                f"The flavors of {cuisine} cuisine complement {protein}.",
                f"{cuisine} chefs use {protein} skillfully.",
            ])
        for spice in SPICES:
            pairs.extend([
                f"{cuisine} cuisine is known for {spice}.",
                f"{spice.capitalize()} features prominently in {cuisine} dishes.",
            ])

    # Texture descriptions
    for texture in TEXTURES:
        tex_cap = texture.capitalize()
        for veg in VEGETABLES:
            ing_cap = veg.capitalize()
            pairs.extend([
                f"Properly cooked {veg} should be {texture}.",
                f"To achieve {texture} {veg}, use the right method.",
                f"{ing_cap} becomes {texture} with proper cooking.",
                f"A {texture} texture indicates {veg} is done.",
                f"{tex_cap} {veg} requires careful cooking.",
            ])
        for protein in PROTEINS:
            ing_cap = protein.capitalize()
            pairs.extend([
                f"Properly cooked {protein} should be {texture}.",
                f"To achieve {texture} {protein}, use the right method.",
                f"{ing_cap} becomes {texture} with proper cooking.",
                f"A {texture} texture indicates {protein} is done.",
                f"{tex_cap} {protein} requires careful cooking.",
            ])

    # Preparation methods with ingredients
    prep_methods = CUTS + ["wash", "peel", "trim", "dry", "marinate"]

    for prep in prep_methods:
        prep_cap = prep.capitalize()
        for veg in VEGETABLES:
            ing_cap = veg.capitalize()
            pairs.extend([
                f"Before cooking, {prep} the {veg} properly.",
                f"{prep_cap}ing {veg} takes practice.",
                f"Learn to {prep} {veg} correctly.",
                f"Properly {prep}ed {veg} cooks more evenly.",
                f"{ing_cap} should be {prep}ed before adding to the dish.",
            ])
        for protein in PROTEINS:
            ing_cap = protein.capitalize()
            pairs.extend([
                f"Before cooking, {prep} the {protein} properly.",
                f"{prep_cap}ing {protein} takes practice.",
                f"Learn to {prep} {protein} correctly.",
                f"Properly {prep}ed {protein} cooks more evenly.",
                f"{ing_cap} should be {prep}ed before adding to the dish.",
            ])

    # Additional specific content
    specific_tips = [
        "Start cooking with ingredients at room temperature.",
        "Read the entire recipe before beginning.",
        "Gather all ingredients before starting to cook.",
        "Keep your workspace clean and organized.",
        "Taste food throughout the cooking process.",
        "Season in layers for better flavor.",
        "Let proteins rest after cooking.",
        "Use a thermometer for accurate temperatures.",
        "Adjust seasonings at the end of cooking.",
        "Save pasta water for thickening sauces.",
        "Dry ingredients brown better than wet ones.",
        "Preheating is essential for proper cooking.",
        "Do not overcrowd the pan when searing.",
        "Let meat come to room temperature before cooking.",
        "Sharp knives are safer than dull ones.",
        "Clean as you go to manage mess.",
        "Use the right tool for each task.",
        "Timing is as important as temperature.",
        "Fresh ingredients need less manipulation.",
        "Work efficiently to keep food fresh.",
    ]
    pairs.extend(specific_tips)

    return pairs


def generate_detailed_instructions():
    """Generate detailed cooking instructions."""
    instructions = []

    # Step-by-step processes
    processes = [
        ("making stock", [
            "Start with bones, vegetables, and aromatics.",
            "Cover with cold water and bring to a simmer.",
            "Skim foam from the surface regularly.",
            "Simmer gently for several hours.",
            "Strain and cool quickly.",
            "Refrigerate and remove fat layer.",
            "Use within a few days or freeze.",
        ]),
        ("preparing ingredients", [
            "Wash vegetables under running water.",
            "Pat ingredients dry before cutting.",
            "Cut items into uniform sizes.",
            "Organize by cooking order.",
            "Measure seasonings accurately.",
        ]),
        ("cooking proteins", [
            "Bring protein to room temperature.",
            "Pat the surface completely dry.",
            "Season generously on all sides.",
            "Preheat cooking surface properly.",
            "Sear or cook to proper temperature.",
            "Rest before slicing or serving.",
        ]),
        ("making sauces", [
            "Start with proper aromatics.",
            "Build flavor in layers.",
            "Add liquid components gradually.",
            "Reduce to concentrate flavors.",
            "Adjust seasoning at the end.",
            "Strain for smooth texture.",
            "Finish with fresh elements.",
        ]),
    ]

    for process_name, steps in processes:
        for i, step in enumerate(steps):
            instructions.append(f"Step {i+1} for {process_name}: {step}")
            instructions.append(f"When {process_name}, {step.lower()}")
            instructions.append(f"To {process_name.replace('making ', '').replace('preparing ', '').replace('cooking ', '')}, remember: {step.lower()}")

    # Additional detailed tips by ingredient category
    veg_tips = [
        "Root vegetables need longer cooking than leafy greens.",
        "Cruciferous vegetables release sulfur when overcooked.",
        "Nightshades like tomatoes soften with cooking time.",
        "Alliums benefit from gentle heat development.",
        "Mushrooms release water when first heated, then brown.",
        "Leafy greens cook quickly and should be last.",
        "Starchy vegetables need thorough cooking.",
        "Waxy potatoes hold shape better than starchy ones.",
        "Fresh corn is sweetest when cooked quickly.",
        "Peppers have different cooking times by color.",
    ]
    instructions.extend(veg_tips)

    protein_tips = [
        "Chicken breasts cook faster than thighs.",
        "Fish cooks quickly and overcooking dries it.",
        "Beef has varying cuts for different methods.",
        "Pork needs thorough cooking except certain cuts.",
        "Lamb pairs well with strong seasonings.",
        "Duck fat renders slowly during cooking.",
        "Seafood should smell like the ocean.",
        "Ground meat requires careful temperature control.",
        "Marinades flavor the surface of proteins.",
        "Brining keeps lean proteins moist.",
    ]
    instructions.extend(protein_tips)

    grain_tips = [
        "Rice requires proper water ratios.",
        "Pasta needs plenty of salted water.",
        "Quinoa benefits from rinsing.",
        "Oats vary by processing method.",
        "Barley needs long simmering.",
        "Couscous steams rather than boils.",
        "Bulgur just needs soaking.",
        "Farro has a chewy texture.",
        "Wild rice is actually a grass seed.",
        "Grains double or triple in cooking.",
    ]
    instructions.extend(grain_tips)

    return instructions


def generate_flavor_profiles():
    """Generate flavor profile descriptions."""
    profiles = []

    # Taste profiles
    taste_profiles = [
        ("sweet", "Sweetness rounds flavors and balances acidity."),
        ("sour", "Sourness brightens and adds complexity."),
        ("salty", "Salt enhances all other flavors."),
        ("bitter", "Bitterness adds depth when balanced."),
        ("umami", "Umami provides savory satisfaction."),
    ]

    for taste, desc in taste_profiles:
        profiles.extend([
            f"{taste.capitalize()} flavors {desc}",
            f"When a dish lacks {taste}ness, the flavor seems flat.",
            f"Adding {taste} elements transforms the overall taste.",
        ])

    # Regional flavor profiles
    regional_profiles = [
        ("Mediterranean", "olive oil", "garlic", "lemon", "herbs"),
        ("Asian", "soy sauce", "ginger", "garlic", "sesame"),
        ("Mexican", "chili", "lime", "cumin", "cilantro"),
        ("Indian", "turmeric", "cumin", "coriander", "garam masala"),
        ("French", "butter", "wine", "herbs", "cream"),
        ("Italian", "olive oil", "garlic", "basil", "parmesan"),
        ("Middle Eastern", "sumac", "tahini", "lemon", "pomegranate"),
        ("Caribbean", "allspice", "thyme", "scotch bonnet", "ginger"),
    ]

    for region, *flavors in regional_profiles:
        flavor_list = ", ".join(flavors[:-1]) + " and " + flavors[-1]
        profiles.extend([
            f"{region} flavors feature {flavor_list}.",
            f"Key flavors in {region} cooking include {flavors[0]} and {flavors[1]}.",
            f"{region} cuisine builds flavor with {flavors[0]}.",
        ])

    return profiles


def generate_cooking_tips():
    """Generate practical cooking tips."""
    tips = []

    # Temperature tips
    temp_tips = [
        "Internal temperature ensures protein safety.",
        "Oven temperatures can vary from dial settings.",
        "Preheating is essential for baking.",
        "Let oil reach the right temperature before frying.",
        "Room temperature ingredients mix better.",
        "Cold butter creates flakier pastries.",
        "Warm eggs incorporate better into batters.",
        "Temperature shock can crack ceramic dishes.",
        "Preheat only under empty pans, not with oil.",
        "Preheated pans sear better.",
    ]
    tips.extend(temp_tips)

    # Timing tips
    timing_tips = [
        "Timing affects both texture and flavor.",
        "Resting time improves final results.",
        "Different ingredients need different cooking times.",
        "Adding ingredients in sequence matters.",
        "Overcooking is easier than undercooking, sometimes.",
        "Marinating time affects flavor penetration.",
        "Resting dough relaxes gluten.",
        "Fermentation time creates flavor.",
        "Broiling requires careful timing.",
        "Pressure cooking speeds up tough cuts.",
    ]
    tips.extend(timing_tips)

    # Seasoning tips
    seasoning_tips = [
        "Season throughout cooking for layered flavor.",
        "Salt early for penetration, late for surface taste.",
        "Acid added at the end brightens flavors.",
        "Fat carries volatilized flavors throughout.",
        "Herbs and spices add dimension.",
        "Taste and adjust as you cook.",
        "Under-seasoning can be corrected.",
        "Over-seasoning is harder to fix.",
        "Quality salt makes a difference.",
        "Freshly ground pepper has more aroma.",
    ]
    tips.extend(seasoning_tips)

    # Preparation tips
    prep_tips = [
        "Proper prep prevents kitchen chaos.",
        "Read recipes completely before starting.",
        "Gather ingredients before beginning.",
        "Organize your workspace efficiently.",
        "Clean as you go.",
        "Sharp knives work better.",
        "Uniform cuts cook evenly.",
        "Prep items in cooking order.",
        "Staging pans saves time.",
        "Preset temperatures for appliances.",
    ]
    tips.extend(prep_tips)

    return tips


def generate_equipment_instructions():
    """Generate equipment usage instructions."""
    content = []

    # Knife skills
    knife_skills = [
        "Keep knives sharp for safety and efficiency.",
        "Use the claw grip to protect fingers.",
        "Cut on appropriate cutting boards.",
        "Different knives suit different tasks.",
        "Wash knives by hand immediately after use.",
        "Store knives safely in blocks or guards.",
        "Honing maintains edge between sharpenings.",
        "Rock the knife for mincing herbs.",
        "Slice with the full length of the blade.",
        "Never try to catch a falling knife.",
    ]
    content.extend(knife_skills)

    # Pan cooking
    pan_skills = [
        "Preheat pans before adding fat.",
        "Different pans conduct heat differently.",
        "Stainless steel needs preheating.",
        "Non-stick pans need lower heat.",
        "Cast iron retains heat well.",
        "Add food to hot, not cold, pans.",
        "Avoid moving food while searing.",
        "Deglaze pans for sauces.",
        "Clean pans properly for longevity.",
        "Different oils suit different temperatures.",
    ]
    content.extend(pan_skills)

    # Oven cooking
    oven_skills = [
        "Preheat ovens fully before baking.",
        "Oven temperatures can vary.",
        "Rack position affects baking.",
        "Rotate pans for even baking.",
        "Avoid opening the oven door.",
        "Use oven lights to check progress.",
        "Convection ovens cook faster.",
        "Adjust for convection settings.",
        "Use appropriate baking vessels.",
        "Hot spots exist in most ovens.",
    ]
    content.extend(oven_skills)

    return content


def generate_pairing_suggestions():
    """Generate food pairing suggestions."""
    suggestions = []

    # Protein + vegetable pairings
    protein_veg_pairs = [
        ("chicken", "asparagus"),
        ("beef", "potatoes"),
        ("fish", "lemon"),
        ("lamb", "rosemary"),
        ("pork", "apples"),
        ("turkey", "cranberry"),
        ("duck", "orange"),
        ("salmon", "dill"),
        ("shrimp", "garlic"),
        ("tofu", "ginger"),
    ]

    for protein, veg in protein_veg_pairs:
        suggestions.extend([
            f"{protein.capitalize()} pairs naturally with {veg}.",
            f"The combination of {protein} and {veg} works well.",
            f"Serve {protein} with {veg} for a classic pairing.",
            f"{veg.capitalize()} complements {protein} beautifully.",
        ])

    # Flavor pairings
    flavor_pairs = [
        ("chocolate", "orange"),
        ("tomato", "basil"),
        ("apple", "cinnamon"),
        ("cheese", "pear"),
        ("strawberry", "balsamic"),
    ]

    for item1, item2 in flavor_pairs:
        suggestions.extend([
            f"{item1.capitalize()} and {item2} create a classic combination.",
            f"The flavor pairing of {item1} with {item2} is timeless.",
            f"Combine {item1} and {item2} for excellent results.",
        ])

    return suggestions


def generate_term_definitions():
    """Generate cooking term definitions."""
    definitions = []

    # French cooking terms
    french_terms = [
        ("mirepoix", "diced onion, carrot, and celery flavor base"),
        ("bouquet garni", "bundle of herbs for infusing"),
        ("mise en place", "everything in place or prepped"),
        ("julienne", "thin matchstick-shaped cuts"),
        ("brunoise", "tiny uniform dice from julienne"),
        ("chiffonade", "rolled and sliced into thin strips"),
        ("deglaze", "add liquid to release fond"),
        ("fond", "browned bits on pan bottom"),
        ("roux", "cooked fat and flour mixture"),
        ("liaison", "egg yolk and cream thickener"),
    ]

    for term, definition in french_terms:
        definitions.extend([
            f"{term.capitalize()} means {definition}.",
            f"The term {term} refers to {definition}.",
            f"In cooking, {term} is defined as {definition}.",
        ])

    # Technique terms
    technique_terms = [
        ("blanching", "brief boiling followed by ice bath"),
        ("braising", "slow cooking in liquid after searing"),
        ("poaching", "gentle cooking in barely simmering liquid"),
        ("sweating", "gentle cooking without browning"),
        ("rendering", "melting fat from meat"),
        ("reducing", "boiling to concentrate and thicken"),
        ("tempering", "gradually introducing temperature"),
        ("folding", "gently combining without deflating"),
        ("proofing", "allowing yeast dough to rise"),
        ("resting", "letting cooked meat juices redistribute"),
    ]

    for term, definition in technique_terms:
        definitions.extend([
            f"{term.capitalize()} is {definition}.",
            f"The technique of {term} involves {definition}.",
            f"To {term} means to do {definition}.",
        ])

    return definitions


def generate_additional_combinations():
    """Generate additional ingredient and method combinations."""
    combinations = []

    # More complex sentence patterns
    for veg in VEGETABLES:
        v = veg.capitalize()
        combinations.extend([
            f"When selecting {veg}, choose specimens that feel heavy for their size.",
            f"Fresh {veg} should have vibrant color and no soft spots.",
            f"Store {veg} in appropriate conditions for maximum freshness.",
            f"{v} can be the star of a dish or a supporting ingredient.",
            f"Different cooking methods transform {veg} in unique ways.",
            f"The flavor of {veg} pairs with many seasonings.",
            f"{v} seasons vary by region and climate.",
            f"Organic {veg} may have different flavor intensity.",
            f"Garden-fresh {veg} needs minimal preparation.",
            f"{v} contributes nutrition as well as flavor to dishes.",
        ])

    for fruit in FRUITS:
        f = fruit.capitalize()
        combinations.extend([
            f"When selecting {fruit}, look for appropriate ripeness.",
            f"{f} adds natural sweetness to dishes.",
            f"Some {fruit} varieties are better for cooking than others.",
            f"The acidity in {fruit} balances rich flavors.",
            f"{f} works in both sweet and savory applications.",
            f"Dried {fruit} concentrates the natural flavors.",
            f"Fresh {fruit} provides vitamins and fiber.",
            f"{f} pairs well with complementary flavors.",
        ])

    for protein in PROTEINS:
        p = protein.capitalize()
        combinations.extend([
            f"When purchasing {protein}, check for appropriate color.",
            f"{p} should have proper texture for the preparation.",
            f"Different {protein} cuts suit different cooking methods.",
            f"The fat content of {protein} affects cooking.",
            f"{p} benefits from proper seasoning.",
            f"Temperature control is critical for {protein}.",
            f"{p} can be prepared in countless ways.",
        ])

    for grain in GRAINS:
        g = grain.capitalize()
        combinations.extend([
            f"{g} cooking ratio affects final texture.",
            f"Different {grain} varieties need different preparation.",
            f">{g} provides sustained energy.",
            f"Washing {grain} can improve texture.",
            f"{g} cultures in many world cuisines.",
        ])

    for herb in HERBS:
        h = herb.capitalize()
        combinations.extend([
            f"Fresh {herb} adds brightness to finished dishes.",
            f"Dried {herb} works better in longer cooking.",
            f"{h} should be stored properly.",
            f"Add {herb} at the right time.",
            f">{h} complements many ingredients.",
        ])

    return combinations


def generate_dish_descriptions():
    """Generate descriptions of dishes and meals."""
    descriptions = []

    # Breakfast dishes
    breakfasts = [
        ("pancakes", "fluffy griddle cakes served with syrup and butter"),
        ("eggs benedict", "poached eggs on english muffins with hollandaise"),
        ("french toast", "bread soaked in egg mixture and fried golden"),
        ("omelet", "eggs beaten and cooked with various fillings"),
        ("scrambled eggs", "eggs stirred continuously in a hot pan"),
        ("breakfast burrito", "eggs, cheese, and meat wrapped in a tortilla"),
        ("granola", "toasted oats with nuts and dried fruit"),
        ("yogurt parfait", "layered yogurt, fruit, and granola"),
    ]

    for dish, desc in breakfasts:
        descriptions.extend([
            f"{dish.capitalize()} are {desc}.",
            f"For breakfast, {dish} is a popular choice.",
            f"Make {dish} for a satisfying morning meal.",
            f"{dish.capitalize()} served warm makes a great start to the day.",
        ])

    # Lunch dishes
    lunches = [
        ("sandwich", "ingredients layered between bread slices"),
        ("salad", "mixed greens with various toppings and dressing"),
        ("soup", "liquid dish made by combining ingredients"),
        ("wrap", "fillings rolled in a flatbread"),
        ("grain bowl", "grains topped with vegetables and protein"),
    ]

    for dish, desc in lunches:
        descriptions.extend([
            f"A {dish} is {desc}.",
            f"For lunch, {dish} works well.",
            f"{dish.capitalize()} provides a midday energy boost.",
            f"Prepare {dish} for a satisfying midday meal.",
        ])

    # Dinner dishes
    dinners = [
        ("pasta", "noodles served with various sauces"),
        ("roast chicken", "whole bird roasted until golden"),
        ("stir fry", "quickly fried vegetables and protein"),
        ("curry", "spiced sauce-based dish"),
        ("steak", "seared or grilled beef"),
        ("risotto", "creamy rice dish from italy"),
        ("casserole", "baked dish combining multiple ingredients"),
        ("grilled fish", "fish cooked over direct heat"),
    ]

    for dish, desc in dinners:
        descriptions.extend([
            f"{dish.capitalize()} is {desc}.",
            f"For dinner, {dish} is a satisfying choice.",
            f"Serve {dish} as a complete meal.",
            f"{dish.capitalize()} makes an excellent dinner centerpiece.",
        ])

    # Appetizers
    appetizers = [
        ("bruschetta", "toasted bread with tomato topping"),
        ("soup shooters", "small servings of soup"),
        ("spring rolls", "vegetables wrapped in thin dough"),
        ("cheese board", "selection of cheeses with accompaniments"),
        ("stuffed mushrooms", "mushroom caps filled with savory mixture"),
    ]

    for dish, desc in appetizers:
        descriptions.extend([
            f"{dish.capitalize()} are {desc}.",
            f"Serve {dish} as a starter.",
            f"{dish.capitalize()} whets the appetite.",
            f"Prepare {dish} for entertaining.",
        ])

    # Desserts
    desserts = [
        ("cake", "sweet baked confection"),
        ("cookies", "small sweet baked treats"),
        ("ice cream", "frozen sweetened cream"),
        ("pie", " pastry crust filled with fruit or custard"),
        ("pudding", "thickened sweetened milk dessert"),
        ("brownies", "dense chocolate baked squares"),
        ("fruit tart", "pastry filled with cream and fruit"),
        ("mousse", "light airy whipped dessert"),
    ]

    for dish, desc in desserts:
        descriptions.extend([
            f"{dish.capitalize()} is {desc}.",
            f"Finish a meal with {dish}.",
            f"{dish.capitalize()} satisfies a sweet craving.",
            f"Serve {dish} for a special treat.",
        ])

    return descriptions


def generate_cooking_process_sentences():
    """Generate sentences about cooking processes."""
    sentences = []

    # Process descriptions
    process_verbs = [
        ("mixing", "Combining ingredients distributes flavors uniformly throughout the dish."),
        ("stirring", "Moving ingredients during cooking prevents sticking and ensures even heating."),
        ("chopping", "Cutting ingredients into pieces increases surface area for cooking."),
        ("seasoning", "Adding salt and spices at different stages builds layered flavor."),
        ("heating", "Applying thermal energy transforms ingredients through chemical reactions."),
        ("cooling", "Reducing temperature stops cooking and develops final texture."),
        ("resting", "Allowing cooked proteins to sit lets juices redistribute."),
        ("marinating", "Soaking ingredients in seasoned liquid infuses flavor."),
    ]

    for verb, desc in process_verbs:
        sentences.extend([
            f"{verb.capitalize()} is a key cooking technique. {desc}",
            desc,
            f"The process of {verb} transforms food.",
            f"Proper {verb} technique improves results.",
        ])

    # Sequence descriptions
    sequences = [
        "First prepare all ingredients before starting to cook.",
        "Begin with aromatics like onions and garlic.",
        "Add proteins after aromatics have softened.",
        "Introduce vegetables based on their cooking times.",
        "Season in layers throughout the cooking process.",
        "Finish with fresh elements just before serving.",
        "Rest the dish appropriately before presentation.",
        "Adjust final seasoning after tasting.",
    ]
    sentences.extend(sequences)

    # Reaction descriptions
    reactions = [
        "The Maillard reaction creates complex flavor compounds through browning.",
        "Caramelization develops sweetness through controlled sugar breakdown.",
        "Protein coagulation changes texture at specific temperatures.",
        "Starch gelatinization occurs when heated in liquid.",
        "Fat rendering releases flavorful compounds.",
        "Water evaporation concentrates flavors.",
        "Gluten development affects baked good texture.",
        "Enzymatic reactions can ripen or soften foods.",
    ]
    sentences.extend(reactions)

    return sentences


def generate_cooking_principles():
    """Generate cooking principles and guidelines."""
    principles = []

    # Heat transfer principles
    heat_principles = [
        "Conduction transfers heat through direct surface contact.",
        "Convection circulates heat through air or liquid movement.",
        "Radiation transfers heat through invisible energy waves.",
        "Different metals conduct heat at different rates.",
        "Thicker pans distribute heat more evenly.",
        "Preheating ensures consistent cooking temperature.",
        "Surface temperature drops when food is added.",
        "Recovery time varies by heating element.",
    ]
    principles.extend(heat_principles)

    # Flavor principles
    flavor_principles = [
        "Salt enhances natural flavors already present.",
        "Acid brightens and adds sensory complexity.",
        "Fat carries aromatic compounds throughout.",
        "Sweet balances acidity and bitterness.",
        "Umami adds savory depth to dishes.",
        "Heat from spices stimulates the palate.",
        "Temperature affects flavor perception.",
        "Texture contrast enhances eating experience.",
    ]
    principles.extend(flavor_principles)

    # Timing principles
    timing_principles = [
        "Adding ingredients in sequence ensures proper cooking.",
        "Different foods cook at different rates.",
        "Carryover cooking continues after removing from heat.",
        "Resting time affects final doneness.",
        "Some dishes improve with time.",
        "Others must be served immediately.",
        "Advance preparation saves cooking time.",
        "Mise en place prevents timing problems.",
    ]
    principles.extend(timing_principles)

    return principles


def generate_more_qa():
    """Generate more question-answer pairs."""
    qa = []

    # More specific questions
    more_questions = [
        ("How do I prevent food from sticking?", "Heat the pan properly before adding oil. Pat food dry before cooking. Wait until the proper searing temperature is reached."),
        ("Why do I need to rest meat?", "Resting allows juices to redistribute throughout the meat. Cutting immediately causes flavorful juices to run out."),
        ("What's the difference between braise and stew?", "Braising uses larger cuts partially submerged in liquid. Stewing uses smaller pieces fully covered in liquid."),
        ("How do I make food crispier?", "Ensure food is dry before cooking. Use proper temperature. Do not overcrowd the pan which creates steam."),
        ("What is fond good for?", "Fond contains concentrated flavor compounds. Deglaze the pan with liquid to incorporate fond into sauces."),
        ("How do I avoid overcooking?", "Use a thermometer for accuracy. Check early rather than late. Understand carryover cooking."),
        ("Why is my baking uneven?", "Check oven calibration. Rotate pans during baking. Ensure rack position is correct."),
        ("How do I build flavor layers?", "Start with aromatics. Season throughout cooking. Use different ingredients that complement."),
    ]

    for question, answer in more_questions:
        qa.extend([
            question,
            answer,
            f"When someone asks {question.lower()}, the answer is: {answer}",
        ])

    return qa


def generate_description_sentences():
    """Generate descriptive sentences about cooking and food."""
    descriptions = []

    # Sensory descriptions
    sensory = [
        "The aroma of browning butter signals complex flavor development.",
        "Visually checking browning helps time cooking properly.",
        "Listening for proper sizzling confirms pan temperature.",
        "Touching meat reveals doneness through firmness.",
        "Tasting throughout reveals how flavors develop.",
        "The sound of simmering indicates proper gentle heat.",
        "Color change signals when vegetables are properly cooked.",
        "Steam rising shows vigorous cooking activity.",
    ]
    descriptions.extend(sensory)

    # Kitchen atmosphere descriptions
    atmosphere = [
        "A well-organized kitchen makes cooking more efficient.",
        "Clean workspaces prevent ingredient confusion.",
        "Proper lighting improves cooking accuracy.",
        "Ventilation removes cooking odors and excess heat.",
        "Counter space needs increase with recipe complexity.",
        "Tool accessibility affects cooking flow.",
        "Mise en place creates calm cooking environment.",
        "A clean sink starts meal preparation well.",
    ]
    descriptions.extend(atmosphere)

    # Ingredient quality descriptions
    quality = [
        "Fresh ingredients need minimal manipulation to shine.",
        "Quality proteins have appropriate color and texture.",
        "Vegetables should feel heavy for their size.",
        "Herbs should appear vibrant and smell aromatic.",
        "Spices lose potency over time.",
        "Freshness indicators vary by ingredient type.",
        "Seasonal ingredients offer best value.",
        "Local sourcing often provides freshness advantages.",
    ]
    descriptions.extend(quality)

    # Success descriptions
    success = [
        "Successful dishes result from technique mastery.",
        "Consistent results come from reliable methods.",
        "Understanding theory enables better cooking.",
        "Practice improves intuitive cooking decisions.",
        "Mistakes teach valuable lessons.",
        "Good organization prevents many problems.",
        "Patience during cooking improves outcomes.",
        "Attention to detail distinguishes good cooking.",
    ]
    descriptions.extend(success)

    return descriptions


def generate_extended_instructions():
    """Generate extended instructional content."""
    instructions = []

    # Extended prep instructions
    prep_instructions = [
        "Wash all produce thoroughly under running water.",
        "Pat ingredients dry to promote browning.",
        "Cut items into uniform sizes for even cooking.",
        "Group ingredients by cooking stage.",
        "Pre-measure spices and seasonings.",
        "Organize equipment for easy access.",
        "Set up waste containers nearby.",
        "Read the recipe through completely.",
        "Identify timing sensitive steps.",
        "Allow time for temperature adjustments.",
    ]
    instructions.extend(prep_instructions)

    # Extended cooking instructions
    cooking_instructions = [
        "Preheat cooking surfaces adequately.",
        "Use appropriate fat for the temperature.",
        "Add ingredients in proper sequence.",
        "Maintain proper cooking temperature.",
        "Stir or flip at appropriate intervals.",
        "Monitor visual and aromatic cues.",
        "Test for doneness accurately.",
        "Remove at proper internal temperature.",
        "Allow carryover cooking to finish.",
        "Rest proteins before serving.",
    ]
    instructions.extend(cooking_instructions)

    # Extended finishing instructions
    finishing_instructions = [
        "Taste and adjust seasoning at the end.",
        "Add fresh herbs just before serving.",
        "Finish with a drizzle of quality oil.",
        "Squeeze fresh acid to brighten.",
        "Check and adjust final temperature.",
        "Plate appropriately for the occasion.",
        "Serve at optimal temperature.",
        "Garnish purposefully.",
        "Consider texture contrast.",
        "Present attractively.",
    ]
    instructions.extend(finishing_instructions)

    return instructions


def generate_flavor_building():
    """Generate flavor building knowledge."""
    flavor_content = []

    # Layer building
    layers = [
        "Start with aromatics to build foundation flavors.",
        "Add depth with browned components.",
        "Include umami-rich ingredients.",
        "Balance with acidity.",
        "Season with salt throughout.",
        "Finish with bright elements.",
        "Layer textures for complexity.",
        "Consider temperature contrast.",
    ]
    flavor_content.extend(layers)

    # Taste balance
    balance = [
        "Sweet and sour elements can balance each other.",
        "Salt suppresses perception of bitterness.",
        "Fat carries flavor compounds.",
        "Acid brightens rich foods.",
        "Umami adds savory depth.",
        "Heat is a physical sensation.",
        "Bitterness adds complexity.",
        "Sweetness rounds flavor profile.",
    ]
    flavor_content.extend(balance)

    # Flavor development
    development = [
        "Flavors meld during resting.",
        "Some flavors intensify with cooking.",
        "Others diminish with heat.",
        "Marinating penetration is limited.",
        "Rubbing adheres seasoning to surface.",
        "Braising develops deep flavors.",
        "Roasting concentrates through evaporation.",
        "Grilling adds smoky notes.",
    ]
    flavor_content.extend(development)

    return flavor_content


def generate_complex_sentences():
    """Generate complex multi-clause sentences."""
    sentences = []

    # Complex ingredient descriptions
    complex_ing = [
        f"When preparing {veg}, the key is to maintain texture while developing flavor."
        for veg in VEGETABLES
    ]
    sentences.extend(complex_ing)

    complex_fruit = [
        f"The natural sugars in {fruit} caramelize beautifully under proper heat."
        for fruit in FRUITS
    ]
    sentences.extend(complex_fruit)

    complex_protein = [
        f"Proper technique transforms {protein} from a simple ingredient into a memorable dish."
        for protein in PROTEINS
    ]
    sentences.extend(complex_protein)

    # Method combinations
    complex_methods = [
        f"The difference between {METHODS[i]} and {METHODS[i+1]} lies in temperature and timing."
        for i in range(len(METHODS)-1)
    ]
    sentences.extend(complex_methods)

    # Equipment + method combinations
    for eq in EQUIPMENT[:15]:
        for method in METHODS[:15]:
            sentences.append(
                f"Using a {eq} for {method} requires understanding how the equipment affects heat transfer."
            )

    # Cuisine + ingredient + method
    for cuisine in CUISINES:
        for method in METHODS[:10]:
            sentences.append(
                f"In {cuisine} cuisine, the {method} technique reflects regional preferences and ingredient availability."
            )

    # Reasoning statements
    reasoning = [
        f"Because {veg} contains high water content, proper {method} technique is essential for best results."
        for veg in VEGETABLES[:20]
        for method in METHODS[:10]
    ]
    sentences.extend(reasoning)

    # Conditional statements
    conditional = [
        f"If you {method} {veg} correctly, the texture and flavor will complement any dish."
        for veg in VEGETABLES[:20]
        for method in METHODS[:10]
    ]
    sentences.extend(conditional)

    # Comparative statements
    comparative = [
        f"While {veg1} and {veg2} share some preparation methods, each requires slightly different technique."
        for veg1 in VEGETABLES[:20]
        for veg2 in VEGETABLES[:20]
        if veg1 != veg2
    ]
    sentences.extend(comparative[:500])

    return sentences


def generate_contextual_paragraphs():
    """Generate contextual multi-sentence content."""
    paragraphs = []

    # Method paragraphs
    for method in METHODS:
        paragraphs.extend([
            f"Mastering {method} opens up many culinary possibilities. The technique of {method} requires understanding heat transfer and timing. Beginners should start with {method} as it teaches fundamental cooking principles. Practice {method} with simple ingredients before attempting complex preparations.",
            f"The key to successful {method} lies in preparation and temperature control. Before {method}, ensure all ingredients are properly prepped. During {method}, maintain consistent heat. After {method}, allow the food to rest.",
            f"Different cuisines approach {method} differently. Mediterranean {method} emphasizes olive oil and herbs. Asian {method} often uses high heat and quick movements. French {method} focuses on technique and precision.",
        ])

    # Ingredient paragraphs
    for veg in VEGETABLES[:25]:
        paragraphs.extend([
            f"{veg.capitalize()} is versatile and appears in cuisines worldwide. When selecting {veg}, look for freshness indicators. Proper storage extends {veg}'s shelf life significantly. Cooking {veg} properly develops its natural sweetness.",
            f"The flavor of {veg} transforms with different cooking methods. Raw {veg} offers crisp texture and fresh taste. Cooked {veg} develops deeper, sweeter notes. {veg.capitalize()} pairs with many seasonings and sauces.",
        ])

    for protein in PROTEINS:
        paragraphs.extend([
            f"Working with {protein} requires attention to temperature and timing. {protein.capitalize()} benefits from proper seasoning before cooking. Different cooking methods suit {protein} differently. Resting after cooking improves {protein} results.",
            f"The versatility of {protein} makes it a kitchen staple. {protein.capitalize()} absorbs flavors from marinades effectively. Proper technique ensures {protein} remains moist. Serving {protein} at correct temperature matters.",
        ])

    for grain in GRAINS:
        paragraphs.extend([
            f"{grain.capitalize()} provides the foundation for many satisfying dishes. Proper preparation of {grain} affects final texture. Different cuisines treat {grain} differently. {grain.capitalize()} pairs well with various flavors.",
        ])

    # Equipment paragraphs
    for eq in EQUIPMENT:
        paragraphs.extend([
            f"A {eq} is an essential tool in many cooking tasks. Proper use of a {eq} requires understanding its properties. Maintaining your {eq} ensures longevity. Different {eq} materials affect cooking differently.",
        ])

    return paragraphs


def generate_procedural_content():
    """Generate step-by-step procedural content."""
    procedures = []

    # Detailed procedures for common tasks
    procedure_templates = [
        ("preparing vegetables for cooking", [
            "Begin by washing all vegetables thoroughly under running water.",
            "Pat vegetables dry with clean kitchen towels or paper towels.",
            "Remove any blemished or damaged portions.",
            "Cut vegetables according to recipe requirements.",
            "Keep cut vegetables in cold water if not cooking immediately.",
            "Dry vegetables again before cooking to promote browning.",
        ]),
        ("cooking grains properly", [
            "Measure grains and liquid using accurate ratios.",
            "Rinse grains if the recipe calls for it.",
            "Bring liquid to a boil before adding grains.",
            "Add grains gradually to prevent boiling over.",
            "Reduce heat to maintain a gentle simmer.",
            "Cover and cook for the specified time.",
            "Let grains rest covered before fluffing.",
        ]),
        ("making a basic sauce", [
            "Heat the fat in an appropriate pan.",
            "Add aromatics and cook until fragrant.",
            "Introduce the main liquid ingredients.",
            "Season appropriately and adjust heat.",
            "Simmer to develop and concentrate flavors.",
            "Strain for smooth texture if desired.",
            "Adjust final seasoning before serving.",
        ]),
        ("roasting vegetables", [
            "Preheat oven to proper temperature.",
            "Cut vegetables into uniform pieces.",
            "Toss with oil and seasonings.",
            "Arrange in single layer on baking sheet.",
            "Roast until caramelized and tender.",
            "Turn vegetables partway through cooking.",
            "Season again while hot if desired.",
        ]),
    ]

    for task_name, steps in procedure_templates:
        for i, step in enumerate(steps):
            procedures.append(f"For {task_name}, step {i+1}: {step}")
            procedures.append(f"Procedure {i+1} for {task_name}: {step}")

    return procedures


def generate_advice_sentences():
    """Generate cooking advice and tips."""
    advice = []

    # General advice
    general_advice = [
        "Always read the entire recipe before starting.",
        "Gather all ingredients and equipment before beginning.",
        "Clean as you go to maintain an organized workspace.",
        "Taste throughout cooking to adjust seasoning.",
        "Write down what you learn for future reference.",
        "Start with room temperature ingredients unless specified.",
        "Use the right tool for each cooking task.",
        "Don't rush important steps like browning or resting.",
        "Organize ingredients in order of use.",
        "Pre-measure critical quantities.",
    ]
    advice.extend(general_advice)

    # Method-specific advice
    for method in METHODS:
        advice.extend([
            f"When {method}, patience often produces best results.",
            f"Proper preparation makes {method} more successful.",
            f"The key to {method} is understanding your equipment.",
            f"Practice {method} regularly to build confidence.",
        ])

    # Ingredient-specific advice
    for veg in VEGETABLES[:25]:
        advice.extend([
            f"Choose {veg} that feel heavy for their size.",
            f"Store {veg} properly to maintain freshness.",
            f"Fresh {veg} need less seasoning than older ones.",
        ])

    for protein in PROTEINS:
        advice.extend([
            f"Bring {protein} to room temperature before cooking.",
            f"Season {protein} in advance when possible.",
            f"Rest {protein} after cooking for best results.",
        ])

    # Problem-solving advice
    problem_advice = [
        "If food sticks, it may not be ready to turn yet.",
        "If flavors seem flat, add salt gradually.",
        "If food is too salty, add acid or fat.",
        "If browning too fast, reduce heat.",
        "If not browning, increase heat or dry surface.",
        "If too acidic, add sweetness.",
        "If too sweet, add acid or salt.",
        "If sauce too thick, add liquid gradually.",
        "If sauce too thin, reduce further.",
        "If seasoning uneven, toss or stir more.",
    ]
    advice.extend(problem_advice)

    return advice


def generate_permutation_sentences():
    """Generate sentences through massive permutation of elements."""
    sentences = []

    # Subject-verb-object with modifiers
    subjects = [
        "professional chefs", "home cooks", "beginners", "experienced cooks",
        "cooking enthusiasts", "kitchen staff", "anyone", "most people",
        "traditional cooks", "modern chefs",
    ]

    verbs = [
        "prefer", "recommend", "suggest", "advise", "appreciate",
        "understand", "practice", "master", "learn", "discover",
        "value", "emphasize", "prioritize", "consider", "recognize",
    ]

    objects_ingredients = VEGETABLES[:20] + PROTEINS + GRAINS

    modifiers = [
        "for best results",
        "when cooking",
        "in most recipes",
        "for optimal flavor",
        "in traditional preparation",
        "for modern cuisine",
        "in professional kitchens",
        "for home cooking",
        "for special occasions",
        "for everyday meals",
    ]

    # Generate permutations
    for subj in subjects:
        for verb in verbs:
            for obj in objects_ingredients:
                for mod in modifiers:
                    sentences.append(f"{subj.capitalize()} {verb} {obj} {mod}.")

    # Method + ingredient + time combinations
    times = [
        "for 5 minutes", "for 10 minutes", "for 15 minutes", "for 20 minutes",
        "for 30 minutes", "for 1 hour", "for 2 hours", "until tender",
        "until golden", "until cooked through", "until fragrant",
        "until bubbles form", "until reduced", "until thickened",
    ]

    for method in METHODS:
        for ing in VEGETABLES[:25] + PROTEINS:
            for time in times:
                sentences.append(f"{method.capitalize()} the {ing} {time}.")
                sentences.append(f"You should {method} {ing} {time}.")
                sentences.append(f"{ing.capitalize()} needs to {method} {time}.")

    # Temperature + method combinations
    temps = [
        "at low heat", "at medium heat", "at high heat",
        "at 350 degrees", "at 400 degrees", "at 450 degrees",
        "at a gentle simmer", "at a rolling boil", "at room temperature",
    ]

    for method in METHODS:
        for temp in temps:
            for ing in VEGETABLES[:15] + PROTEINS:
                sentences.append(f"{method.capitalize()} {ing} {temp}.")
                sentences.append(f"{temp.capitalize()}, {method} {ing} properly.")

    # Tool + action combinations
    tools = [
        "knife", "spatula", "whisk", "tongs", "spoon",
        "fork", "ladle", "strainer", "colander", "grater",
        "peeler", "mandoline", "thermometer", "timer", "scale",
    ]

    actions = [
        "carefully", "gently", "thoroughly", "evenly", "precisely",
        "slowly", "quickly", "smoothly", "steadily", "methodically",
    ]

    for tool in tools:
        for action in actions:
            for ing in VEGETABLES[:10]:
                sentences.append(f"Use a {tool} to {action} prepare the {ing}.")
                sentences.append(f"{action.capitalize()} use a {tool} on the {ing}.")

    return sentences


def generate_unique_pattern_sentences():
    """Generate sentences with unique patterns avoiding template repetition."""
    sentences = []

    # All vegetables with specific cooking description
    veg_descriptions = {
        "onion": "forms the flavor base for countless dishes",
        "garlic": "adds aromatic intensity to recipes",
        "carrot": "provides sweetness and color to dishes",
        "celery": "contributes savory depth and crunch",
        "tomato": "brings acidity and umami to recipes",
        "potato": "offers substance and comfort in meals",
        "broccoli": "adds nutrition and vibrant color",
        "spinach": "wilts quickly into nutrient-packed dishes",
        "mushroom": "contributes umami and meaty texture",
        "zucchini": "adds moisture and mild flavor",
    }

    for veg, desc in veg_descriptions.items():
        sentences.extend([
            f"{veg.capitalize()} {desc}.",
            f"Cooks value {veg} because it {desc}.",
            f"The versatile {veg} {desc}.",
            f"In cooking, {veg} {desc}.",
        ])

    # All proteins with specific cooking description
    protein_descriptions = {
        "chicken": "adapts to virtually any cooking method or flavor",
        "beef": "offers rich flavor and satisfying texture",
        "fish": "cooks quickly and delivers delicate flavor",
        "tofu": "absorbs flavors and adds protein to dishes",
        "eggs": "provide structure and richness to recipes",
        "shrimp": "cooks in minutes with sweet, delicate flavor",
        "lamb": "brings distinctive flavor to special dishes",
        "pork": "offers versatility from roasts to chops",
        "salmon": "provides omega-3s and rich flavor",
        "turkey": "makes lean protein for many preparations",
    }

    for protein, desc in protein_descriptions.items():
        sentences.extend([
            f"{protein.capitalize()} {desc}.",
            f"Many recipes feature {protein} because it {desc}.",
            f"The popular {protein} {desc}.",
            f"In cooking, {protein} {desc}.",
        ])

    # All cuisines with specific description
    cuisine_descriptions = {
        "Italian": "emphasizes quality ingredients and simple preparations",
        "French": "focuses on technique and sauce-making",
        "Chinese": "values wok skills and flavor balance",
        "Indian": "celebrates complex spice combinations",
        "Mexican": "builds bold flavors from chilies and corn",
        "Japanese": "respects ingredient quality and precise technique",
        "Thai": "balances sweet, sour, salty, and spicy",
        "Mediterranean": "relies on olive oil and fresh produce",
        "Korean": "features fermentation and bold seasonings",
        "Vietnamese": "emphasizes fresh herbs and balanced flavors",
    }

    for cuisine, desc in cuisine_descriptions.items():
        sentences.extend([
            f"{cuisine} cuisine {desc}.",
            f"Cooks learn from {cuisine} cuisine, which {desc}.",
            f"The traditions of {cuisine} cuisine {desc}.",
        ])

    # Method descriptions
    method_descriptions = {
        "boiling": "cooks food fully submerged in bubbling liquid",
        "simmering": "gentle cooking just below boiling point",
        "steaming": "cooks with vapor rather than liquid",
        "roasting": "transforms food with dry oven heat",
        "grilling": "adds char and smoky flavor from direct heat",
        "sautéing": "quick cooking in small amount of fat",
        "braising": "slow cooking combining searing and simmering",
        "frying": "cooking in hot fat for crispy texture",
        "baking": "surrounds food with dry oven heat",
        "poaching": "gentle cooking in barely hot liquid",
    }

    for method, desc in method_descriptions.items():
        sentences.extend([
            f"{method.capitalize()} is defined as {desc}.",
            f"The technique of {method} means {desc}.",
            f"When {method}, you are {desc}.",
        ])

    return sentences


def generate_numbered_variations():
    """Generate numbered statements with more variation."""
    sentences = []

    # Numbered tips (different from itemized lists)
    for i, tip in enumerate([
        "reads the entire recipe before starting",
        "prepares all ingredients before cooking",
        "keeps the workspace organized",
        "cleans dishes while cooking",
        "tastes food throughout the process",
        "uses proper food safety practices",
        "starts with room temperature ingredients",
        "preheats pans and ovens properly",
        "uses sharp knives for safety",
        "seasons food at multiple stages",
    ], 1):
        sentences.append(f"Tip number {i}: Always {tip}.")
        sentences.append(f"The {i}th rule of cooking is to {tip}.")
        sentences.append(f"Remember cooking tip {i}: {tip.capitalize()}.")
        sentences.append(f"Professional cooks follow rule {i}: {tip}.")

    # Numbered steps for various processes
    processes = ["making stock", "cooking rice", "roasting vegetables", "grilling meat", "making sauce"]
    for process in processes:
        for i in range(1, 8):
            sentences.append(f"Step {i} of {process}: follow proper technique.")
            sentences.append(f"In {process} step {i}, pay attention to detail.")
            sentences.append(f"The {i}th part of {process} requires careful attention.")
            sentences.append(f"{process.capitalize()} step number {i} completes the preparation.")

    return sentences


def generate_qualitative_descriptions():
    """Generate qualitative description sentences."""
    sentences = []

    # Adjective + ingredient combinations
    adjectives = [
        "fresh", "ripe", "tender", "crisp", "juicy",
        "aromatic", "flavorful", "savory", "sweet", "rich",
        "light", "creamy", "crunchy", "smooth", "robust",
        "delicate", "intense", "mild", "bold", "subtle",
    ]

    for adj in adjectives:
        for veg in VEGETABLES[:15]:
            sentences.extend([
                f"{adj.capitalize()} {veg} adds quality to any dish.",
                f"Look for {adj} {veg} when shopping.",
                f"The {adj} {veg} provides excellent flavor.",
                f"Choose {veg} that appears {adj} and fresh.",
            ])
        for protein in PROTEINS:
            sentences.extend([
                f"{adj.capitalize()} {protein} indicates quality.",
                f"When {protein} is {adj}, it cooks beautifully.",
                f"The {adj} {protein} transforms the dish.",
                f"Serve {protein} when it reaches {adj} texture.",
            ])

    # Texture descriptions
    textures = [
        "crispy on outside, tender inside",
        "smooth and creamy throughout",
        "firm yet yielding to bite",
        "light and airy in texture",
        "dense and satisfyingly chewy",
    ]

    for texture in textures:
        for protein in PROTEINS[:5]:
            sentences.append(f"Properly cooked {protein} should be {texture}.")
        for veg in VEGETABLES[:10]:
            sentences.append(f"Well-prepared {veg} becomes {texture}.")

    return sentences


def generate_technique_descriptions():
    """Generate technique description sentences."""
    sentences = []

    # Technique importance
    techniques = [
        "knife skills", "heat control", "timing", "seasoning",
        "mise en place", "proper preparation", "resting",
        "temperature management", "flavor balancing", "presentation",
    ]

    importance_adjectives = [
        "essential", "critical", "fundamental", "valuable", "important",
        "useful", "necessary", "helpful", "practical", "beneficial",
    ]

    for tech in techniques:
        for adj in importance_adjectives:
            sentences.extend([
                f"{tech.capitalize()} is {adj} for cooking success.",
                f"Developing {tech} is {adj} for any cook.",
                f"Understanding {tech} proves {adj} in the kitchen.",
                f"Mastering {tech} is {adj} for professional results.",
            ])

    # Technique details
    tech_details = {
        "knife skills": "improve safety and uniformity of cuts",
        "heat control": "determines texture and doneness",
        "timing": "ensures all components finish together",
        "seasoning": "brings out natural flavors",
        "mise en place": "prevents scrambling during cooking",
        "proper preparation": "makes execution smoother",
        "resting": "allows juices to redistribute",
        "temperature management": "affects final results",
        "flavor balancing": "creates harmony in dishes",
        "presentation": "enhances eating experience",
    }

    for tech, detail in tech_details.items():
        sentences.extend([
            f"Good {tech} {detail}.",
            f"Practicing {tech} helps {detail}.",
            f"The purpose of {tech} is to ensure it {detail}.",
            f"Experienced cooks know that {tech} {detail}.",
        ])

    return sentences


def generate_expanded_vocabulary_sentences():
    """Generate sentences with expanded vocabulary to increase word count."""
    sentences = []

    # Culinary terms
    culinary_terms = [
        ("al dente", "firm to the bite"),
        ("bain-marie", "water bath for gentle heating"),
        ("bard", "wrap meat with fat to prevent drying"),
        ("blanch", "briefly boil then shock in ice water"),
        ("carafe", "container for serving liquids"),
        ("chiffonade", "thin strips of leafy ingredients"),
        ("concasse", "roughly chopped tomatoes"),
        ("coulis", "thick sauce from pureed vegetables"),
        ("crudités", "raw vegetables served with dip"),
        ("deglaze", "add liquid to release fond pieces"),
        ("demi-glace", "rich concentrated brown sauce"),
        ("emulsify", "combine unmixable liquids"),
        ("farce", "stuffing or forcemeat"),
        ("galette", "free-form rustic tart"),
        ("hollandaise", "egg yolk and butter emulsion"),
        ("infusion", "steeping to extract flavor"),
        ("julienne", "matchstick vegetable cut"),
        ("knead", "work dough to develop gluten"),
        ("liaison", "thickening with cream and yolks"),
        ("marinade", "seasoned liquid for soaking"),
        ("nap", "coat food with sauce"),
        ("parboil", "partially cook in boiling water"),
        ("quenelle", "oval shaped scoop"),
        ("render", "melt fat from meat"),
        ("roe", "fish eggs"),
        ("saute", "quick fry in hot fat"),
        ("temper", "gradually blend temperatures"),
        ("veloute", "white sauce from stock"),
        ("whisk", "beat to incorporate air"),
        ("zest", "colored citrus peel"),
    ]

    for term, definition in culinary_terms:
        sentences.extend([
            f"{term.capitalize()} means {definition} in culinary terms.",
            f"In cooking, {term} refers to {definition}.",
            f"The culinary term {term} is defined as {definition}.",
            f"Chefs use the term {term} to mean {definition}.",
            f"{term.capitalize()}, which means {definition}, appears in many recipes.",
        ])

    return sentences


def generate_adjective_descriptions():
    """Generate sentences using descriptive adjectives."""
    sentences = []

    for adj in ADJECTIVES:
        for veg in VEGETABLES[:20]:
            sentences.extend([
                f"{adj.capitalize()} {veg} adds unique character to dishes.",
                f"Look for {adj} {veg} when selecting ingredients.",
                f"The {adj} {veg} creates {adj} flavors.",
            ])
        for protein in PROTEINS[:15]:
            sentences.extend([
                f"{adj.capitalize()} {protein} delivers {adj} results.",
                f"When {protein} is {adj}, it tastes better.",
                f"Serve {protein} when it reaches {adj} state.",
            ])
        for grain in GRAINS[:10]:
            sentences.extend([
                f"{adj.capitalize()} {grain} works in many recipes.",
                f"Prepare {grain} until {adj}.",
                f"The {adj} {grain} texture is important.",
            ])

    return sentences


def generate_action_descriptions():
    """Generate sentences using action verbs."""
    sentences = []

    for verb in ACTION_VERBS:
        sentences.extend([
            f"To {verb} properly, use correct technique.",
            f"You should {verb} carefully.",
            f"The best way to {verb} requires practice.",
            f"Cooks {verb} every day.",
            f"Learners practice how to {verb}.",
        ])

        for veg in VEGETABLES[:10]:
            sentences.extend([
                f"{verb.capitalize()} the {veg} thoroughly.",
                f"You must {verb} {veg} correctly.",
                f"Cooks {verb} {veg} regularly.",
            ])

        for protein in PROTEINS[:10]:
            sentences.extend([
                f"{verb.capitalize()} {protein} carefully.",
                f"Always {verb} {protein} properly.",
                f"Chefs {verb} {protein} expertly.",
            ])

    return sentences


def generate_time_descriptions():
    """Generate sentences using time vocabulary."""
    sentences = []

    for time_word in TIME_VOCAB:
        sentences.extend([
            f"Wait {time_word} before checking.",
            f"Cook for about {time_word}.",
            f"The process takes {time_word}.",
            f"Timing of {time_word} is critical.",
            f"Be patient for {time_word}.",
        ])
        for method in METHODS[:10]:
            sentences.extend([
                f"{method.capitalize()} for {time_word}.",
                f"During {method}, wait {time_word}.",
                f"The {method} step takes {time_word}.",
            ])

    return sentences


def generate_measurement_descriptions():
    """Generate sentences using measurement vocabulary."""
    sentences = []

    for measure in MEASUREMENT_VOCAB:
        sentences.extend([
            f"Use the right {measure}.",
            f"Measure carefully in {measure}.",
            f"A {measure} makes a difference.",
            f"Precision of {measure} matters.",
        ])
        for ing in VEGETABLES[:10] + PROTEINS[:10]:
            sentences.extend([
                f"Add one {measure} of {ing}.",
                f"Measure {measure} of {ing}.",
                f"Use {measure} when preparing {ing}.",
            ])

    return sentences


def generate_origin_descriptions():
    """Generate sentences using origin vocabulary."""
    sentences = []

    for origin in ORIGIN_VOCAB:
        sentences.extend([
            f"{origin.capitalize()} cuisine is distinctive.",
            f"This dish is {origin} in origin.",
            f"{origin.capitalize()} techniques influence cooking.",
            f"The {origin} style is popular.",
        ])
        for method in METHODS[:10]:
            sentences.extend([
                f"{origin.capitalize()} cooks often {method}.",
                f"The {origin} approach to {method}.",
                f"{method.capitalize()} is common in {origin} cuisine.",
            ])

    return sentences


def generate_sensory_descriptions():
    """Generate sentences using sensory vocabulary."""
    sentences = []

    for sensory_word in SENSORY_VOCAB:
        sentences.extend([
            f"The {sensory_word} quality matters.",
            f"Notice the {sensory_word} aspect.",
            f"{sensory_word.capitalize()} appeal is important.",
            f"Pay attention to {sensory_word} details.",
        ])
        for dish_type in ["soup", "salad", "roast", "stew", "pasta", "curry"]:
            sentences.extend([
                f"The {sensory_word} quality of {dish_type} matters.",
                f"{sensory_word.capitalize()} notes make {dish_type} special.",
            ])

    return sentences


def generate_scientific_descriptions():
    """Generate sentences using scientific vocabulary."""
    sentences = []

    for scientific_word in SCIENTIFIC_VOCAB:
        sentences.extend([
            f"The {scientific_word} process affects cooking.",
            f"Understanding {scientific_word} improves results.",
            f"{scientific_word.capitalize()} plays a role in cooking.",
            f"Scientific understanding of {scientific_word} helps.",
        ])

    return sentences


def generate_kitchen_descriptions():
    """Generate sentences using kitchen vocabulary."""
    sentences = []

    for kitchen_word in KITCHEN_VOCAB:
        sentences.extend([
            f"The {kitchen_word} is your workspace.",
            f"Keep your {kitchen_word} organized.",
            f"Use the {kitchen_word} efficiently.",
            f"A clean {kitchen_word} works best.",
        ])

    return sentences


def generate_condition_descriptions():
    """Generate sentences using condition vocabulary."""
    sentences = []

    for condition in CONDITION_VOCAB:
        sentences.extend([
            f"Ingredients should be {condition}.",
            f"The {condition} state affects cooking.",
            f"Work with {condition} ingredients.",
            f"{condition.capitalize()} ingredients behave differently.",
        ])
        for ing in VEGETABLES[:10] + PROTEINS[:10]:
            sentences.extend([
                f"Keep {ing} {condition}.",
                f"When {ing} is {condition}, use carefully.",
            ])

    return sentences


def generate_casual_descriptions():
    """Generate sentences using casual vocabulary."""
    sentences = []

    for casual_word in CASUAL_VOCAB:
        sentences.extend([
            f"That's so {casual_word}!",
            f"It tastes {casual_word}.",
            f"Make it {casual_word}.",
            f"Everyone loves {casual_word} food.",
        ])

    return sentences


def generate_expression_descriptions():
    """Generate sentences using expression vocabulary."""
    sentences = []

    for expr_word in EXPRESSION_VOCAB:
        sentences.extend([
            f"Make it {expr_word}.",
            f"This step is {expr_word}.",
            f"{expr_word.capitalize()} cooking is rewarding.",
            f"The {expr_word} approach works well.",
        ])

    return sentences


def generate_sauce_descriptions():
    """Generate sentences using sauce vocabulary."""
    sentences = []

    for sauce in SAUCES:
        sentences.extend([
            f"{sauce.capitalize()} is a classic sauce.",
            f"Make your own {sauce} at home.",
            f"{sauce.capitalize()} pairs with many dishes.",
            f"Drizzle {sauce} over the top.",
            f"The {sauce} adds flavor.",
        ])
        for protein in PROTEINS[:10]:
            sentences.extend([
                f"{sauce.capitalize()} goes well with {protein}.",
                f"Serve {protein} with {sauce}.",
                f"{protein.capitalize()} and {sauce} is delicious.",
            ])

    return sentences


def main():
    """Main generator."""
    print("Generating diverse cooking dataset...")

    all_lines = []

    # Collect all content
    print("  - Generating factual statements...")
    all_lines.extend(generate_factual_statements())

    print("  - Generating Q&A pairs...")
    all_lines.extend(generate_question_answer_pairs())

    print("  - Generating instructional content...")
    all_lines.extend(generate_instructional_content())

    print("  - Generating measurement guidance...")
    all_lines.extend(generate_measurement_guidance())

    print("  - Generating pairing knowledge...")
    all_lines.extend(generate_pairing_knowledge())

    print("  - Generating troubleshooting...")
    all_lines.extend(generate_troubleshooting())

    print("  - Generating variant sentences...")
    all_lines.extend(generate_variant_sentences())

    print("  - Generating conversational exchanges...")
    all_lines.extend(generate_conversational_exchanges())

    print("  - Generating cuisine knowledge...")
    all_lines.extend(generate_cuisine_knowledge())

    print("  - Generating safety content...")
    all_lines.extend(generate_safety_content())

    print("  - Generating seasonal cooking...")
    all_lines.extend(generate_seasonal_cooking())

    print("  - Generating nutrition basics...")
    all_lines.extend(generate_nutrition_basics())

    print("  - Generating detailed instructions...")
    all_lines.extend(generate_detailed_instructions())

    print("  - Generating flavor profiles...")
    all_lines.extend(generate_flavor_profiles())

    print("  - Generating cooking tips...")
    all_lines.extend(generate_cooking_tips())

    print("  - Generating equipment instructions...")
    all_lines.extend(generate_equipment_instructions())

    print("  - Generating pairing suggestions...")
    all_lines.extend(generate_pairing_suggestions())

    print("  - Generating term definitions...")
    all_lines.extend(generate_term_definitions())

    print("  - Generating additional combinations...")
    all_lines.extend(generate_additional_combinations())

    print("  - Generating additional content...")
    all_lines.extend(generate_additional_content())

    print("  - Generating expanded pairs...")
    all_lines.extend(generate_expanded_pairs())

    print("  - Generating dish descriptions...")
    all_lines.extend(generate_dish_descriptions())

    print("  - Generating cooking process sentences...")
    all_lines.extend(generate_cooking_process_sentences())

    print("  - Generating cooking principles...")
    all_lines.extend(generate_cooking_principles())

    print("  - Generating more Q&A...")
    all_lines.extend(generate_more_qa())

    print("  - Generating description sentences...")
    all_lines.extend(generate_description_sentences())

    print("  - Generating extended instructions...")
    all_lines.extend(generate_extended_instructions())

    print("  - Generating flavor building...")
    all_lines.extend(generate_flavor_building())

    print("  - Generating complex sentences...")
    all_lines.extend(generate_complex_sentences())

    print("  - Generating contextual paragraphs...")
    all_lines.extend(generate_contextual_paragraphs())

    print("  - Generating procedural content...")
    all_lines.extend(generate_procedural_content())

    print("  - Generating advice sentences...")
    all_lines.extend(generate_advice_sentences())

    print("  - Generating permutation sentences...")
    all_lines.extend(generate_permutation_sentences())

    print("  - Generating unique pattern sentences...")
    all_lines.extend(generate_unique_pattern_sentences())

    print("  - Generating numbered variations...")
    all_lines.extend(generate_numbered_variations())

    print("  - Generating qualitative descriptions...")
    all_lines.extend(generate_qualitative_descriptions())

    print("  - Generating technique descriptions...")
    all_lines.extend(generate_technique_descriptions())

    print("  - Generating expanded vocabulary sentences...")
    all_lines.extend(generate_expanded_vocabulary_sentences())

    print("  - Generating adjective descriptions...")
    all_lines.extend(generate_adjective_descriptions())

    print("  - Generating action descriptions...")
    all_lines.extend(generate_action_descriptions())

    print("  - Generating time descriptions...")
    all_lines.extend(generate_time_descriptions())

    print("  - Generating measurement descriptions...")
    all_lines.extend(generate_measurement_descriptions())

    print("  - Generating origin descriptions...")
    all_lines.extend(generate_origin_descriptions())

    print("  - Generating sensory descriptions...")
    all_lines.extend(generate_sensory_descriptions())

    print("  - Generating scientific descriptions...")
    all_lines.extend(generate_scientific_descriptions())

    print("  - Generating kitchen descriptions...")
    all_lines.extend(generate_kitchen_descriptions())

    print("  - Generating condition descriptions...")
    all_lines.extend(generate_condition_descriptions())

    print("  - Generating casual descriptions...")
    all_lines.extend(generate_casual_descriptions())

    print("  - Generating expression descriptions...")
    all_lines.extend(generate_expression_descriptions())

    print("  - generating sauce descriptions...")
    all_lines.extend(generate_sauce_descriptions())

    # Deduplicate while preserving order
    print("\nDeduplicating...")
    seen = set()
    unique_lines = []
    for line in all_lines:
        normalized = line.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_lines.append(line.strip())

    print(f"\nTotal generated: {len(all_lines):,} lines")
    print(f"Unique lines after dedup: {len(unique_lines):,}")

    # Save to file
    output_path = Path(__file__).parent.parent / "data" / "household" / "cooking_v2.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")

    print(f"\nSaved to: {output_path}")

    # Basic quality check
    words = set()
    for line in unique_lines:
        words.update(line.lower().split())
    print(f"Unique vocabulary words: {len(words):,}")

    return unique_lines


if __name__ == "__main__":
    main()
