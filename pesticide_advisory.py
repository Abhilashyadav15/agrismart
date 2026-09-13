"""
AgriSmart - Pesticide Recommendation & Advisory Module
Provides targeted, resource-saving pesticide and bio-control recommendations
based on crop type, observed pest/disease, and severity.
"""

PESTICIDE_DATABASE = {
    "Rice": {
        "Blast (Magnaporthe oryzae)": {
            "symptoms": "Diamond-shaped lesions on leaves with grey/white centers and brown margins.",
            "chemical_solution": "Tricyclazole 75% WP @ 0.6 g/liter or Isoprothiolane 40% EC @ 1.5 ml/liter.",
            "biological_alternative": "Pseudomonas fluorescens (10 g/kg seed treatment or 2.5 kg/ha foliar spray).",
            "dosage": "120-150 grams/acre diluted in 200 liters of water.",
            "resource_saving_tip": "Apply spot spraying only on affected patches in early morning. Avoid blanket spraying to reduce chemical runoff by 45%."
        },
        "Stem Borer (Scirpophaga incertulas)": {
            "symptoms": "Dead heart during vegetative stage; white head (empty panicles) during flowering.",
            "chemical_solution": "Chlorantraniliprole 18.5% SC @ 0.3 ml/liter or Cartap Hydrochloride 4G @ 10 kg/acre.",
            "biological_alternative": "Trichogramma japonicum egg parasitoid cards (5 cards/acre) + Pheromone traps (8/acre).",
            "dosage": "60 ml/acre diluted in 200 liters of water.",
            "resource_saving_tip": "Deploy pheromone traps to monitor pest threshold (2 moths/trap/day) before applying any chemical spray."
        },
        "Brown Planthopper (Nilaparvata lugens)": {
            "symptoms": "Hopper burn: Circular yellowing patches that turn brown and dry up rapidly.",
            "chemical_solution": "Pymetrozine 50% WDG @ 0.6 g/liter or Dinotefuran 20% SG @ 0.4 g/liter.",
            "biological_alternative": "Neem seed kernel extract (NSKE 5%) or Neem Oil (1500 ppm) @ 3 ml/liter.",
            "dosage": "120 grams/acre directed specifically to the base of the plant.",
            "resource_saving_tip": "Direct spray nozzle to plant base where hoppers congregate. Alternate drying and wetting of fields to disrupt breeding."
        }
    },
    "Maize": {
        "Fall Armyworm (Spodoptera frugiperda)": {
            "symptoms": "Ragged feeding holes on whorl leaves and large amounts of sawdust-like frass.",
            "chemical_solution": "Emamectin Benzoate 5% SG @ 0.4 g/liter or Chlorantraniliprole 18.5% SC @ 0.4 ml/liter.",
            "biological_alternative": "Bacillus thuringiensis (Bt) kurstaki @ 2 g/liter or Metarhizium anisopliae @ 5 g/liter.",
            "dosage": "80 grams/acre applied directly into the leaf whorls.",
            "resource_saving_tip": "Apply directly into the central whorl using a knapsack sprayer without nozzle for targeted placement, cutting wastage by 50%."
        },
        "Leaf Blight (Exserohilum turcicum)": {
            "symptoms": "Long, elliptical, grayish-green or tan lesions on lower leaves progressing upward.",
            "chemical_solution": "Mancozeb 75% WP @ 2.5 g/liter or Azoxystrobin + Difenoconazole @ 1 ml/liter.",
            "biological_alternative": "Trichoderma harzianum foliar spray @ 5 g/liter.",
            "dosage": "500 grams/acre in 200 liters of water.",
            "resource_saving_tip": "Initiate treatment upon appearance of initial lesions on lower canopy only."
        }
    },
    "Cotton": {
        "Pink Bollworm (Pectinophora gossypiella)": {
            "symptoms": "Rosetted flowers, exit holes in green bolls, premature boll opening with stained fiber.",
            "chemical_solution": "Profenofos 50% EC @ 2 ml/liter or Spinetoram 11.7% SC @ 0.8 ml/liter.",
            "biological_alternative": "Install Gossyplure pheromone traps (5/acre) + Trichogramma bactrae @ 60,000 eggs/acre.",
            "dosage": "400 ml/acre diluted in 200 liters of water.",
            "resource_saving_tip": "Strictly time spraying 45-60 days after sowing when moth trap catches exceed 8 moths/trap/night."
        },
        "Whitefly & Aphids": {
            "symptoms": "Curling of leaves, honeydew secretion, sooty mold growth, stunted plant vigor.",
            "chemical_solution": "Diafenthiuron 50% WP @ 1.2 g/liter or Flonicamid 50% WG @ 0.4 g/liter.",
            "biological_alternative": "Verticillium lecanii @ 5 g/liter + Yellow sticky traps (15 traps/acre).",
            "dosage": "240 grams/acre in 200 liters of water.",
            "resource_saving_tip": "Use yellow sticky traps for non-chemical mass trapping to delay or eliminate chemical interventions."
        }
    },
    "Wheat": {
        "Yellow / Stripe Rust (Puccinia striiformis)": {
            "symptoms": "Yellow-orange pustules arranged in linear stripes along leaf veins.",
            "chemical_solution": "Propiconazole 25% EC @ 1 ml/liter or Tebuconazole 25.9% EC @ 1 ml/liter.",
            "biological_alternative": "Trichoderma viride spray @ 5 g/liter.",
            "dosage": "200 ml/acre in 200 liters of water.",
            "resource_saving_tip": "Monitor cold, humid weather spells and spray once at first sign of stripe rust to stop field-wide transmission."
        },
        "Aphids (Rhopalosiphum padi)": {
            "symptoms": "Clusters of small green/black bugs on spikes and flag leaves sucking sap.",
            "chemical_solution": "Thiamethoxam 25% WG @ 0.3 g/liter or Imidacloprid 17.8% SL @ 0.5 ml/liter.",
            "biological_alternative": "Neem Oil (1500 ppm) @ 3 ml/liter + encourage ladybird beetle predators.",
            "dosage": "50 grams/acre in 150 liters of water.",
            "resource_saving_tip": "Only treat if aphid population exceeds economic threshold level (ETL) of 10-15 aphids/earhead."
        }
    },
    "Chickpea": {
        "Pod Borer (Helicoverpa armigera)": {
            "symptoms": "Round holes bored into pods with caterpillars feeding inside.",
            "chemical_solution": "Indoxacarb 14.5% SC @ 1 ml/liter or Chlorantraniliprole 18.5% SC @ 0.3 ml/liter.",
            "biological_alternative": "HaNPV (Helicoverpa Nuclear Polyhedrosis Virus) @ 100 LE/acre + Bird perches (15/acre).",
            "dosage": "150 ml/acre in 200 liters of water.",
            "resource_saving_tip": "Install bird perches across field so predatory birds hunt larvae naturally without pesticide costs."
        },
        "Fusarium Wilt (Fusarium oxysporum)": {
            "symptoms": "Sudden drooping of leaves, internal vascular discoloration (brown xylem).",
            "chemical_solution": "Carbendazim 50% WP @ 2 g/liter (soil drenching).",
            "biological_alternative": "Seed treatment with Trichoderma viride @ 4 g/kg seed + soil application with FYM.",
            "dosage": "Seed treatment: 2 g/kg; Drenching: 300 g/acre.",
            "resource_saving_tip": "Prioritize preventive biological seed treatment, avoiding expensive post-infection chemical drenching."
        }
    },
    "Sugarcane": {
        "Early Shoot Borer (Chilo infuscatellus)": {
            "symptoms": "Dead hearts in young shoots that emit foul odor when pulled out.",
            "chemical_solution": "Fipronil 0.3% G @ 10 kg/acre or Chlorantraniliprole 0.4% G @ 7.5 kg/acre.",
            "biological_alternative": "Release Trichogramma chilonis @ 20,000/acre at 10-day intervals.",
            "dosage": "Apply granules at the base of shoots and cover with light irrigation.",
            "resource_saving_tip": "Practice light earthing up of cane rows at 35-45 days to physically restrict larvae entry."
        },
        "Red Rot (Colletotrichum falcatum)": {
            "symptoms": "Discoloration of third/fourth leaf, longitudinal reddening of internal stalk with white cross bands.",
            "chemical_solution": "Carbendazim 50% WP @ 1 g/liter (sett soaking before planting).",
            "biological_alternative": "Trichoderma harzianum fortified farmyard manure application @ 2.5 kg/acre.",
            "dosage": "Sett dip: 100 grams in 100 liters of water.",
            "resource_saving_tip": "Soak setts prior to planting rather than field spraying, reducing active ingredient volume by over 80%."
        }
    }
}


def get_supported_crops():
    """Returns list of crops supported by the pesticide advisory module."""
    return list(PESTICIDE_DATABASE.keys())


def get_diseases_for_crop(crop: str):
    """Returns list of common diseases/pests for a given crop."""
    return list(PESTICIDE_DATABASE.get(crop, {}).keys())


def get_recommendation(crop: str, disease: str) -> dict:
    """
    Returns structured pesticide, biological alternative, dosage, and
    resource optimization guidance.
    """
    crop_data = PESTICIDE_DATABASE.get(crop, {})
    recommendation = crop_data.get(disease, None)
    if not recommendation:
        return {
            "found": False,
            "message": "Specific advisory not found for this combination. Consult your local Krishi Vigyan Kendra (KVK)."
        }
    
    return {
        "found": True,
        "crop": crop,
        "disease": disease,
        **recommendation
    }
