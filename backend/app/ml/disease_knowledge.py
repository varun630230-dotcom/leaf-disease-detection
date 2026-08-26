"""LeafGuard AI — Structured Agronomic & Disease Knowledge Base.

Provides evidence-based plant pathology information for supported conditions:
- Pathogen & Causes
- Visual Symptoms & Indicators
- Agronomic Effects & Impact
- Recommended Immediate Actions
- Long-Term Prevention & Integrated Pest Management (IPM)
- Environmental Risk Factors
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class DiseaseGuide(BaseModel):
    plant: str
    condition: str
    is_healthy: bool
    pathogen_type: Optional[str] = None
    causative_agent: Optional[str] = None
    symptoms: List[str] = []
    effects: List[str] = []
    immediate_actions: List[str] = []
    prevention_measures: List[str] = []
    risk_factors: List[str] = []
    source: str = "University Extension & FAO Plant Protection Guidelines"


# Comprehensive agronomic knowledge base for PlantVillage classes
DISEASE_KNOWLEDGE: Dict[str, dict] = {
    # ── Tomato Diseases ───────────────────────────────────────────
    "Tomato Late Blight": {
        "plant": "Tomato",
        "condition": "Late Blight",
        "is_healthy": False,
        "pathogen_type": "Oomycete (Fungus-like organism)",
        "causative_agent": "Phytophthora infestans",
        "symptoms": [
            "Irregular water-soaked greasy lesions on leaves and stems",
            "Pale green to chlorotic halo surrounding expanding dark necrotic spots",
            "White, fuzzy fungal sporulation on undersides of leaves during high humidity",
            "Rapid browning and collapse of entire leaf sections within 48-72 hours"
        ],
        "effects": [
            "Severe canopy defoliation drastically reducing photosynthetic capacity",
            "Systemic vine collapse and dark brown girdling lesions on main stems",
            "Destructive greasy brown rot on developing and mature tomato fruits",
            "Potential total crop loss under persistent damp conditions"
        ],
        "immediate_actions": [
            "Immediately prune and safely destroy severely infected foliage (do not compost)",
            "Cease all overhead sprinkler irrigation; apply water strictly at soil base",
            "Improve field drainage and stake vines to keep foliage off moist soil",
            "Sanitize all pruning shears and tools with 70% alcohol between plants"
        ],
        "prevention_measures": [
            "Plant certified disease-free transplants and blight-resistant cultivars",
            "Maintain wide plant spacing (60-90 cm) to ensure maximum canopy airflow",
            "Apply protective bio-fungicide or copper-based sprays prior to prolonged rain",
            "Implement strict 3-year crop rotation avoiding potatoes and solanaceous crops"
        ],
        "risk_factors": [
            "Cool to moderate temperatures (15°C - 22°C / 60°F - 72°F)",
            "Prolonged leaf wetness (>6 hours) from rain, fog, or heavy morning dew",
            "High relative ambient humidity exceeding 90%"
        ]
    },
    "Tomato Early Blight": {
        "plant": "Tomato",
        "condition": "Early Blight",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Alternaria solani",
        "symptoms": [
            "Distinctive dark brown to black spots with concentric target-like rings",
            "Yellow chlorotic halos surrounding lesions, typically starting on lower leaves",
            "Progressive upward defoliation from ground level toward the upper canopy",
            "Dark, sunken, leathery cankers near the base of the stems"
        ],
        "effects": [
            "Premature loss of lower foliage exposing fruits to sunscald",
            "Reduced fruit yield and stunted plant vigor",
            "Stem lesions causing seedling collar rot and lodging"
        ],
        "immediate_actions": [
            "Remove lower infected leaves touching or near the soil line",
            "Apply a thick organic mulch (straw, woodchips) around the base to prevent soil splash",
            "Avoid working in the field when foliage is wet to prevent spore dissemination"
        ],
        "prevention_measures": [
            "Stake and cage plants to promote vertical growth and rapid leaf drying",
            "Practice 2-3 year crop rotation away from solanaceous plants",
            "Apply preventative broad-spectrum bio-fungicide at early flowering stage"
        ],
        "risk_factors": [
            "Warm temperatures (24°C - 29°C / 75°F - 85°F)",
            "Frequent alternating wet and dry weather cycles",
            "Nutrient-deficient soil, particularly low nitrogen stress"
        ]
    },
    "Tomato Bacterial Spot": {
        "plant": "Tomato",
        "condition": "Bacterial Spot",
        "is_healthy": False,
        "pathogen_type": "Bacterial Pathogen",
        "causative_agent": "Xanthomonas campestris pv. vesicatoria",
        "symptoms": [
            "Small (1-3 mm), dark, greasy, water-soaked spots on leaves",
            "Lesions turning black with prominent yellow borders and ragged centers",
            "Leaves turning yellow, blighted, and dropping prematurely",
            "Small, raised, scab-like blister spots on green fruit surfaces"
        ],
        "effects": [
            "Substantial leaf drop leading to severe fruit sunscald",
            "Superficial fruit scabs making tomatoes unmarketable",
            "Secondary rot infections entering through fruit lesions"
        ],
        "immediate_actions": [
            "Avoid overhead irrigation; water exclusively via drip or furrow systems",
            "Do not cultivate, weed, or harvest plants while foliage is wet",
            "Promptly discard severely infected plants to prevent field-wide spread"
        ],
        "prevention_measures": [
            "Use certified hot-water-treated seeds and certified clean transplants",
            "Apply preventative copper-mancozeb or Bacillus-based bactericide sprays",
            "Eradicate nightshade weeds and solanaceous volunteer plants nearby"
        ],
        "risk_factors": [
            "Warm temperatures (24°C - 30°C)",
            "Wind-driven rain, sprinkler irrigation, and high humidity (>85%)"
        ]
    },
    "Tomato Septoria Leaf Spot": {
        "plant": "Tomato",
        "condition": "Septoria Leaf Spot",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Septoria lycopersici",
        "symptoms": [
            "Numerous small circular spots (2-5 mm) with greyish-white centers and dark brown margins",
            "Tiny black specks (pycnidia fruiting bodies) visible inside the center of lesions",
            "Lower leaves yellowing and dropping sequentially upward"
        ],
        "effects": [
            "Severe progressive defoliation from base to top",
            "Fruit exposure to intense sunlight and sunscald",
            "Weakened plant architecture and diminished overall yield"
        ],
        "immediate_actions": [
            "Prune off affected lower leaves showing dense circular spotting",
            "Apply clean straw mulch to block fungal spores splashing from soil",
            "Disinfect pruning tools between plants"
        ],
        "prevention_measures": [
            "Rotate crops annually with non-host crops (corn, legumes, brassicas)",
            "Ensure 75 cm plant spacing for rapid canopy evaporation",
            "Apply preventative copper or chlorothalonil fungicide before canopy closure"
        ],
        "risk_factors": [
            "Extended wet periods at temperatures between 20°C - 25°C"
        ]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "plant": "Tomato",
        "condition": "Yellow Leaf Curl Virus",
        "is_healthy": False,
        "pathogen_type": "Viral Pathogen (Begomovirus)",
        "causative_agent": "Tomato Yellow Leaf Curl Virus (TYLCV), vectored by Whiteflies",
        "symptoms": [
            "Severe upward curling and cupping of leaflet margins",
            "Marked interveinal yellowing (chlorosis) on new growth",
            "Stunted, bushy, erect plant growth with abnormally small leaves",
            "Excessive flower drop with failure to set fruit"
        ],
        "effects": [
            "Up to 100% yield loss if infection occurs before fruit set",
            "Stunted unmarketable fruits with uneven ripening"
        ],
        "immediate_actions": [
            "Rogue and bag infected plants immediately to prevent whitefly vector spread",
            "Deploy yellow sticky cards throughout the greenhouse/field to monitor whiteflies",
            "Apply insecticidal soap, neem oil, or selective whitefly controls"
        ],
        "prevention_measures": [
            "Plant TYLCV-resistant or tolerant hybrid varieties",
            "Install 50-mesh insect screening in greenhouse structures",
            "Maintain a 2-month host-free crop break between planting seasons"
        ],
        "risk_factors": [
            "High populations of silverleaf whitefly (Bemisia tabaci)",
            "Warm, dry, windy weather favoring insect migration"
        ]
    },
    "Tomato Healthy": {
        "plant": "Tomato",
        "condition": "Healthy",
        "is_healthy": True,
        "symptoms": [
            "Vibrant deep-green foliage with uniform surface coloration",
            "Normal pinnate leaf venation without chlorosis or necrosis",
            "Absence of fungal sporulation, bacterial water-soaking, or viral curling"
        ],
        "effects": [
            "Optimal photosynthetic rate supporting robust vegetative and fruit growth",
            "High disease resistance and vigorous fruit set"
        ],
        "immediate_actions": [
            "Maintain current watering schedule (1-2 inches per week at root level)",
            "Ensure balanced N-P-K fertilization with adequate calcium to prevent blossom end rot"
        ],
        "prevention_measures": [
            "Continue scouting weekly for early signs of pests or lesions",
            "Maintain mulch barrier and stake vines off the soil surface",
            "Avoid working in field rows when leaves are damp"
        ],
        "risk_factors": [
            "High humidity and prolonged leaf wetness could invite spore germination"
        ]
    },

    # ── Potato Diseases ───────────────────────────────────────────
    "Potato Early Blight": {
        "plant": "Potato",
        "condition": "Early Blight",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Alternaria solani",
        "symptoms": [
            "Dark brown concentric target-pattern spots on older leaves",
            "Yellow chlorotic halos surrounding lesions with leaf margin dieback",
            "Dark sunken lesions on potato tuber skins with dry rot underneath"
        ],
        "effects": [
            "Premature canopy dieback reducing tuber bulking and final yield",
            "Storage tuber rot causing significant post-harvest losses"
        ],
        "immediate_actions": [
            "Prune infected bottom leaves",
            "Ensure uniform soil moisture to reduce plant physiological stress",
            "Apply protective copper or bio-fungicide sprays"
        ],
        "prevention_measures": [
            "Plant certified high-vigor seed potatoes",
            "Maintain adequate soil nitrogen and potassium levels throughout tuber filling",
            "Allow tubers to fully mature and skin-set before harvesting"
        ],
        "risk_factors": [
            "Alternating wet and dry periods, warm temperatures (24°C - 30°C)"
        ]
    },
    "Potato Late Blight": {
        "plant": "Potato",
        "condition": "Late Blight",
        "is_healthy": False,
        "pathogen_type": "Oomycete Pathogen",
        "causative_agent": "Phytophthora infestans",
        "symptoms": [
            "Rapidly expanding water-soaked dark lesions on leaf tips and margins",
            "White mildew growth on leaf undersides in morning humidity",
            "Foul-smelling rapid vine breakdown and tuber rot"
        ],
        "effects": [
            "Catastrophic vine collapse within days; destructive tuber rot in storage"
        ],
        "immediate_actions": [
            "Destroy infected foliage; apply targeted late-blight fungicides immediately",
            "Do not harvest wet tubers to avoid inoculating the entire storage batch"
        ],
        "prevention_measures": [
            "Use certified disease-free seed tubers and resistant varieties",
            "Hill up soil well around potato hills to shield tubers from washed spores"
        ],
        "risk_factors": [
            "Cool, humid, overcast weather with frequent rain (15°C - 20°C)"
        ]
    },

    # ── Apple Diseases ────────────────────────────────────────────
    "Apple Apple Scab": {
        "plant": "Apple",
        "condition": "Apple Scab",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Venturia inaequalis",
        "symptoms": [
            "Olive-green to velvety dark brown spots on upper leaf surfaces",
            "Leaves becoming distorted, puckered, and dropping prematurely in mid-summer",
            "Corky, scabby brown lesions on apple fruit skin, causing cracking"
        ],
        "effects": [
            "Severe defoliation weakening tree reserves for subsequent seasons",
            "Deformed, cracked, unmarketable fruit"
        ],
        "immediate_actions": [
            "Rake and shred or compost fallen autumn leaves where fungus overwinters",
            "Prune tree canopy during dormancy to maximize sunlight penetration and airflow"
        ],
        "prevention_measures": [
            "Plant scab-resistant apple cultivars (e.g. Liberty, Enterprise, Freedom)",
            "Apply preventative sulfur or copper sprays from green tip through petal fall"
        ],
        "risk_factors": [
            "Spring rains with temperatures between 13°C - 24°C"
        ]
    },
    "Apple Black Rot": {
        "plant": "Apple",
        "condition": "Black Rot (Frog-Eye Leaf Spot)",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Botryosphaeria obtusa",
        "symptoms": [
            "Frog-eye leaf spots: small purple specks expanding into brown spots with purple margins",
            "Firm, concentric dark brown to black rot on ripening fruit",
            "Sunken cankers on limbs and branches"
        ],
        "effects": [
            "Limb dieback, leaf loss, and complete rot of infected fruit before harvest"
        ],
        "immediate_actions": [
            "Prune out dead wood, fire blight strikes, and cankered branches 15 cm below damage",
            "Remove all mummified fruit hanging on the tree or lying on the ground"
        ],
        "prevention_measures": [
            "Maintain balanced tree nutrition and minimize bark injury",
            "Apply protective fungicide sprays starting at bloom"
        ],
        "risk_factors": [
            "Warm wet weather (20°C - 27°C) and unmanaged dead orchard wood"
        ]
    },
    "Apple Cedar Apple Rust": {
        "plant": "Apple",
        "condition": "Cedar Apple Rust",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen (Gymnosporangium)",
        "causative_agent": "Gymnosporangium juniperi-virginianae",
        "symptoms": [
            "Bright yellow-orange spots on upper leaf surface with red margins",
            "Small raised tube-like fungal structures on leaf undersides in late spring"
        ],
        "effects": [
            "Early leaf drop and blemished fruit; requires nearby juniper/cedar host"
        ],
        "immediate_actions": [
            "Remove nearby Eastern red cedar or juniper trees within 500 meters if feasible",
            "Prune galls from ornamental junipers in late winter"
        ],
        "prevention_measures": [
            "Choose rust-immune apple varieties",
            "Apply targeted fungicide at pink bud stage through bloom"
        ],
        "risk_factors": [
            "Proximity to infected juniper/cedar trees and warm spring rains"
        ]
    },

    # ── Corn (Maize) Diseases ─────────────────────────────────────
    "Corn Common Rust": {
        "plant": "Corn",
        "condition": "Common Rust",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Puccinia sorghi",
        "symptoms": [
            "Golden-brown to cinnamon-brown powdery pustules scattered on both leaf surfaces",
            "Pustules rupturing leaf epidermis, turning brownish-black late in the season"
        ],
        "effects": [
            "Reduced photosynthetic active area, grain fill impairment, and stalk lodging"
        ],
        "immediate_actions": [
            "Monitor pustule density on lower and middle ear leaves",
            "Consider fungicide application if disease reaches upper canopy before dent stage"
        ],
        "prevention_measures": [
            "Plant resistant hybrid corn varieties carrying Rp gene resistance",
            "Plant early to avoid peak mid-summer airborne spore showers"
        ],
        "risk_factors": [
            "Cool to moderate temperatures (16°C - 25°C) and high relative humidity"
        ]
    },
    "Corn Northern Leaf Blight": {
        "plant": "Corn",
        "condition": "Northern Leaf Blight",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Exserohilum turcicum",
        "symptoms": [
            "Long, elliptical, cigar-shaped greyish-green to tan lesions (2.5 - 15 cm long)",
            "Dark sporulation appearing in lesions during damp mornings"
        ],
        "effects": [
            "Premature death of ear leaves causing major grain yield reduction"
        ],
        "immediate_actions": [
            "Assess threshold: spray foliar fungicide if lesions appear on ear leaf before silking"
        ],
        "prevention_measures": [
            "Rotate with non-host crops (soybean, alfalfa)",
            "Till under residue to accelerate fungal decomposition"
        ],
        "risk_factors": [
            "Moderate temperatures (18°C - 27°C) and extended dew periods (>6 hours)"
        ]
    },

    # ── Grape Diseases ────────────────────────────────────────────
    "Grape Black Rot": {
        "plant": "Grape",
        "condition": "Black Rot",
        "is_healthy": False,
        "pathogen_type": "Fungal Pathogen",
        "causative_agent": "Guignardia bidwellii",
        "symptoms": [
            "Small reddish-brown circular spots on leaves with dark borders and black pycnidia",
            "Berries turning pale, then shriveling into hard, wrinkled, black mummies"
        ],
        "effects": [
            "Complete destruction of grape clusters; up to 80% vineyard crop loss"
        ],
        "immediate_actions": [
            "Remove and destroy all mummified grape clusters from the trellis during pruning",
            "Prune canopy shoots to expose clusters to sunlight and wind"
        ],
        "prevention_measures": [
            "Implement early-season fungicide program from early shoot growth through 4 weeks post-bloom",
            "Maintain weed-free zone beneath grape trellises"
        ],
        "risk_factors": [
            "Warm wet weather (21°C - 27°C) with persistent leaf wetness"
        ]
    },

    # ── General Fallback ──────────────────────────────────────────
    "General Disease": {
        "plant": "Supported Crop",
        "condition": "Foliar Plant Pathology",
        "is_healthy": False,
        "pathogen_type": "Foliar Pathogen",
        "symptoms": [
            "Localized necrotic lesions or chlorotic margins",
            "Surface discoloration deviating from healthy green leaf tissue"
        ],
        "effects": [
            "Reduced photosynthetic activity and potential defoliation if left unmanaged"
        ],
        "immediate_actions": [
            "Prune and dispose of symptomatic leaf tissue",
            "Water at the base of the plant; avoid wetting the canopy",
            "Improve air circulation around the plant canopy"
        ],
        "prevention_measures": [
            "Practice annual crop rotation and maintain balanced fertilization",
            "Monitor surrounding crops weekly for early disease onset"
        ],
        "risk_factors": [
            "High relative humidity, excessive moisture, and poor ventilation"
        ]
    }
}


def get_disease_guide(plant: str, disease: Optional[str], is_healthy: bool) -> DiseaseGuide:
    """Retrieve structured agronomic information for a specific plant/condition."""
    if is_healthy:
        healthy_key = f"{plant} Healthy"
        if healthy_key in DISEASE_KNOWLEDGE:
            return DiseaseGuide(**DISEASE_KNOWLEDGE[healthy_key])
        return DiseaseGuide(
            plant=plant,
            condition="Healthy",
            is_healthy=True,
            symptoms=["Normal uniform green coloration with intact leaf morphology"],
            effects=["Optimal photosynthetic efficiency and vegetative growth"],
            immediate_actions=["Maintain regular drip irrigation and nutrient schedule"],
            prevention_measures=["Scout weekly and maintain good garden sanitation"],
            risk_factors=["Prolonged wetness may create favorable pathogen conditions"],
        )

    # Diseased condition lookup
    disease_clean = disease or "Condition"
    lookup_key = f"{plant} {disease_clean}"

    if lookup_key in DISEASE_KNOWLEDGE:
        return DiseaseGuide(**DISEASE_KNOWLEDGE[lookup_key])

    # Search for partial match on disease name
    for key, data in DISEASE_KNOWLEDGE.items():
        if disease_clean.lower() in key.lower() and plant.lower() in key.lower():
            return DiseaseGuide(**data)

    # Search by disease only
    for key, data in DISEASE_KNOWLEDGE.items():
        if disease_clean.lower() in key.lower():
            guide_copy = data.copy()
            guide_copy["plant"] = plant
            return DiseaseGuide(**guide_copy)

    # Fallback to General Disease
    general = DISEASE_KNOWLEDGE["General Disease"].copy()
    general["plant"] = plant
    general["condition"] = disease_clean
    return DiseaseGuide(**general)
