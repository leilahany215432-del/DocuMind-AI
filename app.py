from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.database import add_document, add_question, recent_documents, recent_questions
from src.document_engine import (
    answer_with_local_model,
    answer_with_optional_llm,
    build_index,
    extract_entities,
    extract_pages,
    local_summary,
    search_chunks,
    split_chunks,
)

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
.block-container { padding-top: 2rem; }
.hero { background: linear-gradient(135deg,#0f766e,#155e75); color:#fff; padding: 1.5rem 2rem; border-radius: 18px; margin-bottom: 1.2rem; }
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin: .4rem 0 0; opacity: .92; }
.source { background:#f0fdfa; border-right: 4px solid #0f766e; padding: .8rem 1rem; border-radius: 8px; margin:.45rem 0; }
.small { color:#55716d; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>DocuMind AI</h1><p>افهمي مستنداتك بسرعة — استخراج، تلخيص، أسئلة، ومصادر موثّقة داخل الملف</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("إعدادات المنصة")
    st.caption("نسخة MVP محلية — لا يتم إرسال الملفات للخارج إلا عند تفعيل نموذج اختياري في ملف .env")
    use_llm = st.toggle("استخدام النموذج الخارجي إن كان مفعّلًا", value=True)
    st.divider()
    st.markdown("**الأنواع المدعومة**\n\nPDF · DOCX · TXT · MD")

if "document" not in st.session_state:
    st.session_state.document = None

uploaded = st.file_uploader("ارفعي مستندًا للتحليل", type=["pdf", "docx", "txt", "md"], help="يمكن رفع ملف واحد في كل مرة في هذه النسخة الأولية.")

if uploaded:
    try:
        pages = extract_pages(uploaded.name, uploaded.getvalue())
        chunks = split_chunks(pages, uploaded.name)
        if not chunks:
            st.error("لم أستطع استخراج نص من هذا الملف. تأكدي أنه غير ممسوح ضوئيًا أو استخدمي OCR أولًا.")
        else:
            vectorizer, matrix = build_index(chunks)
            document_id = add_document(uploaded.name, Path(uploaded.name).suffix.upper(), len(pages) if uploaded.name.lower().endswith('.pdf') else None, sum(len(c.text) for c in chunks))
            st.session_state.document = {"name": uploaded.name, "chunks": chunks, "vectorizer": vectorizer, "matrix": matrix, "id": document_id, "pages": pages}
            st.success(f"تمت فهرسة {len(chunks)} مقطعًا من {uploaded.name}")
    except Exception as exc:
        st.error(f"حدث خطأ أثناء قراءة الملف: {exc}")

doc = st.session_state.document
if not doc:
    st.info("ارفعي ملفًا من الأعلى للبدء. بعد الرفع ستظهر هنا لوحة التحليل والأسئلة والمصادر.")
else:
    chunks, vectorizer, matrix = doc["chunks"], doc["vectorizer"], doc["matrix"]
    tabs = st.tabs(["نظرة عامة", "اسألي المستند", "استخراج البيانات", "النص المستخرج", "السجل"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("المقاطع المفهرسة", len(chunks))
        c2.metric("عدد الصفحات", len(doc["pages"]) if doc["pages"] else "—")
        c3.metric("عدد الأحرف", f"{sum(len(c.text) for c in chunks):,}")
        st.subheader("ملخص تلقائي")
        if st.button("إنشاء الملخص", type="primary"):
            with st.spinner("أحلل محتوى المستند..."):
                summary = None
                if use_llm:
                    summary = answer_with_optional_llm("لخّص المستند في نقاط واضحة مع الحفاظ على أهم الأرقام والتواريخ.", local_summary(chunks, vectorizer, matrix, 8))
                if not summary:
                    selected = local_summary(chunks, vectorizer, matrix, 5)
                    summary = "\n\n".join(f"• {c.text}" for c, _ in selected)
                st.session_state.summary = summary
        if st.session_state.get("summary"):
            st.write(st.session_state.summary)

    with tabs[1]:
        st.subheader("اسألي عن محتوى الملف")
        question = st.text_input("اكتبي سؤالك", placeholder="مثال: ما أهم التواريخ أو الالتزامات المذكورة؟")
        top_k = st.slider("عدد المصادر المستخدمة", 1, 8, 5)
        if st.button("إجابة", type="primary", disabled=not question.strip()):
            results = search_chunks(question, chunks, vectorizer, matrix, top_k)
            answer = answer_with_optional_llm(question, results) if use_llm else None
            answer = answer or answer_with_local_model(question, results)
            add_question(doc["id"], question, answer)
            st.session_state.last_answer = answer
            st.session_state.last_results = results
        if st.session_state.get("last_answer"):
            st.markdown("### الإجابة")
            st.write(st.session_state.last_answer)
            st.markdown("### المصادر داخل المستند")
            for i, (chunk, score) in enumerate(st.session_state.get("last_results", []), start=1):
                location = f"صفحة {chunk.page}" if chunk.page else "موضع داخل المستند"
                st.markdown(f'<div class="source"><b>المصدر {i} — {html.escape(chunk.source)} · {location}</b><br>{html.escape(chunk.text)}<br><span class="small">درجة الصلة: {score:.2f}</span></div>', unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("البيانات المهمة المستخرجة")
        all_text = "\n".join(c.text for c in chunks)
        entities = extract_entities(all_text)
        for label, values in entities.items():
            st.markdown(f"**{label}**")
            if values:
                st.write("، ".join(values))
            else:
                st.caption("لم يتم العثور على عناصر واضحة")

    with tabs[3]:
        st.subheader("النص المستخرج")
        for idx, (page_text, page_number) in enumerate(doc["pages"], start=1):
            title = f"الصفحة {page_number}" if page_number else f"المحتوى — القسم {idx}"
            with st.expander(title, expanded=idx == 1):
                st.text_area("", page_text, height=220, label_visibility="collapsed")

    with tabs[4]:
        st.subheader("السجل المحلي")
        st.markdown("#### المستندات")
        for row in recent_documents():
            st.write(f"{row['name']} — {row['file_type']} — {row['created_at']}")
        st.markdown("#### آخر الأسئلة")
        for row in recent_questions():
            st.write(f"**{row['question']}** — {row['name']} — {row['created_at']}")
