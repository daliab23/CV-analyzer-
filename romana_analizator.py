import json
import re
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("ro_core_news_lg")


def load_and_build_matcher(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    for domain, content in taxonomy.items():
        # Procesare Skills
        skills_data = content.get("Skills", {})
        if isinstance(skills_data, dict):
            for cat, s_list in skills_data.items():
                matcher.add(f"SKILL|{domain}", [nlp.make_doc(s) for s in s_list])
        else:
            matcher.add(f"SKILL|{domain}", [nlp.make_doc(s) for s in skills_data])

        # Procesare Roles și Education
        matcher.add(f"ROLE|{domain}", [nlp.make_doc(r) for r in content.get("Roles", [])])
        matcher.add(f"EDU|{domain}", [nlp.make_doc(e) for e in content.get("Education", [])])

    return matcher

mathcher = load_and_build_matcher("data_romana.json")

def is_required(token):
    context_verbs = {
        "colabora", "susține", "asista", "raporta",
        "lucra", "ajuta", "coordona", "sprijini"
    }

    for ancestor in token.ancestors:
        if ancestor.lemma_.lower() in context_verbs:
            return False
    return True

def extract_entities(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    entities = {"skills": set(), "roles": set(), "education": set()}

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        type_ = label.split("|")[0]     #skill or role
        span = doc[start:end]

        # Verificăm contextul gramatical (ex: "Am lucrat cu un Python Developer" -> ignoră)
        if type_ in ["SKILL", "ROLE"] and not is_required(span.root):
            continue

        if type_ == "SKILL":
            entities["skills"].add(span.text.lower())
        elif type_ == "ROLE":
            entities["roles"].add(span.text.lower())
        elif type_ == "EDU":
            entities["education"].add(span.text.lower())

    return entities



def final_score(cv_text, job_text, matcher):
    cv_ent = extract_entities(cv_text, matcher)
    job_ent = extract_entities(job_text, matcher)

    # Scorul semantic (spaCy similarity)
    sem_score = nlp(cv_text).similarity(nlp(job_text))


    def get_overlap(cv_set, job_set):
        if not job_set: return 1.0, set()
        return len(cv_set & job_set) / len(job_set), job_set - cv_set

    s_score, m_skills = get_overlap(cv_ent["skills"], job_ent["skills"])
    r_score, m_roles = get_overlap(cv_ent["roles"], job_ent["roles"])
    e_score, m_edu = get_overlap(cv_ent["education"], job_ent["education"])

    def get_years(t):
        match = re.search(r"(\d+)\+?\s*(?:ani|an)", t.lower())
        return int(match.group(1)) if match else 0

    cv_y, job_y = get_years(cv_text), get_years(job_text)
    y_score = 1.0 if cv_y >= job_y or job_y == 0 else cv_y / job_y

    total = (0.25 * sem_score) + (0.3 * s_score) + (0.3 * r_score) + (0.05 * e_score) + (0.1 * y_score)

    return {
        "scor_final": round(total * 100, 2),
        "abilitati_lipsa": list(m_skills),
        "roluri_lipsa": list(m_roles),
        "experienta_detectata": f"{cv_y} ani"
    }



