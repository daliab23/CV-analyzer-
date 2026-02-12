import romana_analizator as ra
import curatare_text as ctx

matcher = ra.load_and_build_matcher("data_romana.json")


print("CV QA TESTER vs JOB INGINER ML")
cv_text = ctx.extract_word_text("cv_qa_tester.docx")
job_text = ctx.extract_word_text("job_ml_inginer.docx")
print(ra.final_score(cv_text, job_text, matcher))

print("CV QA TESTER vs JOB QA TESTER")
cv_text = ctx.extract_word_text("cv_qa_tester.docx")
job_text = ctx.extract_word_text("job_qa_tester.docx")
print(ra.final_score(cv_text, job_text, matcher))

print("CV Finante vs JOB Finante")
cv_text = ctx.extract_word_text("cv_finante.docx")
job_text = ctx.extract_word_text("job_finante.docx")
print(ra.final_score(cv_text, job_text, matcher))