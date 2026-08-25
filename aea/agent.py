import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from case_jobs.integrations.openai_client import get_openai_client


logger = logging.getLogger(__name__)

GENERATION_MODEL = "gpt-5.6"


@dataclass(frozen=True)
class DocumentPrompt:
    name: str
    template: str


def format_current_date() -> str:
    return datetime.now().strftime("%B %d, %Y").replace(" 0", " ")


def build_prompt_registry():
    """Build the prompt registry for each L1a output document."""
    # Get today’s date in the desired format
    current_date = format_current_date()

    return {
        "Petition Cover Letter": DocumentPrompt(
            name="Petition Cover Letter",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a petition cover letter for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Application for Change of Status (Form I-539) and Application for Employment Authorization Document (Form I-765)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **Principal Applicant:** [Insert Full Name of Principal Applicant]
                
                Dear Sir/Madam:
                
                Please accept the enclosed I-539 application to change the status of [Applicant's Full Name] to [Dependent Status] dependent status. [Applicant's] current [Status] will expire on [MM/DD/YYYY].
                
                The following documents are enclosed in support of this application:
                
                1. Two separate checks in the amount of:  
                   - $[Amount] for the Form I-539 application filing fee  
                   - $[Amount] for the Form I-765 application filing fee  
                2. Form G-1145, *e-Notification of Application/Petition Acceptance*  
                3. Form G-28, *Notice of Entry of Appearance as Attorney or Accredited Representative*  
                4. Form I-539, *Application to Change/Extend Nonimmigrant Status*  
                5. Form I-765, *Application for Employment Authorization*, with two passport-style photos attached  
                6. Documentation for the principal applicant ([Spouse's Full Name]):  
                   - Form I-797A Approval Notice for Form I-539, Petition for a Nonimmigrant Worker  
                   - Passport biographic page  
                7. Documentation for the dependent applicant ([Applicant's Full Name]):  
                   - Form I-94 and CBP travel history  
                   - F-1 Visa  
                   - Form I-20  
                   - Passport biographic page  
                   - Marriage certificate  
                
                We respectfully request the prompt adjudication of this application. Thank you for your attention to this matter.

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**    
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
        ),
        "Eligibility Memorandum": DocumentPrompt(
            name="Eligibility Memorandum",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Eligibility Memorandum for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Eligibility Memorandum in Support of Application for Employment Authorization (Form I-765)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **Eligibility Category:** [Insert Category, e.g., (c)(9), (c)(3)(C), etc.]  
                
                Dear Sir/Madam:  
                
                Please find enclosed this Eligibility Memorandum in support of [Applicant’s Full Name]’s Form I-765, *Application for Employment Authorization*. This memorandum outlines the applicable eligibility category, summarizes the factual background, and identifies the legal basis for the requested employment authorization.  
                
                **Background**  
                [Provide a concise description of the applicant’s current immigration status, class of admission, and any pending or approved applications/petitions related to the eligibility category. Include receipt numbers if available and only if present in the provided file.]  
                
                **Eligibility Basis**  
                [State the exact eligibility category (e.g., (c)(9) – pending adjustment of status; (c)(3)(C) – post-completion OPT; (c)(8) – asylum applicant, etc.) and briefly explain how the evidence enclosed meets the regulatory requirements under 8 CFR 274a.12.]  
                
                **Enclosures in Support of This Memorandum**  
                The following documents are provided in support of this memorandum and the corresponding Form I-765:  
                
                1. Form I-765, *Application for Employment Authorization*  
                2. Form G-1145, *E-Notification of Application/Petition Acceptance* (if provided)  
                3. Form G-28, *Notice of Entry of Appearance as Attorney* (if represented)  
                4. Two passport-style photos (per USCIS specifications)  
                5. Form I-94, *Arrival/Departure Record*  
                6. Copy of passport biographic page and U.S. visa (if available)  
                7. Prior EAD cards (front and back), if applicable  
                8. Category-specific supporting evidence (e.g., Form I-485/I-589/I-20 with recommendation, marriage certificate, USCIS notices)  
                
                We respectfully submit this memorandum and the enclosed supporting documentation to establish [Applicant’s Full Name]’s eligibility for the requested employment authorization. We request favorable adjudication at your earliest convenience.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_
                **[Attorney/Representative Name], [Title]**  
                **[Firm/Organization Name]**  

                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
        ),
        "Evidence-Organization Chart": DocumentPrompt(
            name="Evidence-Organization Chart",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Evidence-Organization Chart for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Evidence – Organization Chart in Support of Application for Employment Authorization (Form I-765)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **Eligibility Category:** [Insert Category, e.g., (c)(9), (c)(3)(C), etc.]  
                
                Dear Sir/Madam:  
                
                Please find enclosed the organizational chart of [Company/Organization Name] in support of [Applicant’s Full Name]’s Form I-765, *Application for Employment Authorization*. This document is submitted to demonstrate the applicant’s position within the company, the reporting structure, and the relationship to other personnel relevant to the eligibility category asserted.  
                
                **Purpose of Submission**  
                The organizational chart provides a clear visual representation of:  
                - The applicant’s current or proposed role within the company.  
                - The chain of command and supervisory relationships.  
                - How the applicant’s position aligns with the company’s operational structure.  
                - The relationship between the applicant’s position and other departments or personnel relevant to the application.  
                
                **Enclosures in Support of This Evidence**  
                1. Company organizational chart identifying the applicant’s position.  
                2. Legend or key explaining titles, reporting lines, and departments (if not clearly marked on the chart).  
                3. Supporting documentation (if available in the provided file set), such as:  
                   - Job description corresponding to the position shown.  
                   - Employer letter confirming the role and reporting structure.  
                   - Any relevant USCIS notices, contracts, or internal HR records.  
                
                We respectfully submit this organizational chart and accompanying evidence to assist USCIS in understanding [Applicant’s Full Name]’s position and its relation to the eligibility requirements for employment authorization.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_ 
                **[Attorney/Representative Name], [Title]**  
                **[Firm/Organization Name]**  

                ```

                Step 3.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 4.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 5.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 6.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file .
                Step 7. Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
                ),
            ),
            "Exhibit List": DocumentPrompt(
                name="Exhibit List",
                template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **COMPANY LETTERHEAD**
                
                **List of Supporting Documents**  
                **TN Visa Application**

                **Petitioner:** [COMPANY NAME]  
                **Beneficiary:** [BENEFICIARY’S NAME]

                Pursuant to the United States-Mexico-Canada Agreement, a foreign national is entitled to enter the United States under the TN Status Visa category. Below is a complete list of supporting documents submitted to establish that **Mr. / Ms. [BENEFICIARY’S NAME]**, a citizen of **[Canada / Mexico]**, is qualified for the TN Visa.

                | **Exhibit 1** | **Forms & Fees** |
                |---------------|------------------|
                |               | 1.1 Cover Letter |
                |               | 1.2 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.3 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.4 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.5 Form G-28 “Notice of Entry of Appearance as Attorney” |
                |               | 1.6 Form I-907 “Request for Premium Processing” |
                |               | 1.7 Form I-129 + TN Supplement “Petition for a Nonimmigrant Worker” |
                | **Exhibit 2** | **Proof of Nationality & Lawful Stay in the U.S. and Dependents** |
                |---------------|------------------------------------------------------------------|
                |               | 2.1 [BENEFICIARY’S NAME] – Passport |
                |               | 2.2 [BENEFICIARY’S NAME] – Previous Visa |
                |               | 2.3 [BENEFICIARY’S NAME] – Birth Certificate |
                |               | 2.4 [BENEFICIARY’S NAME] – Resume |
                |               | 2.5 [BENEFICIARY’S NAME] – Educational Credentials |
                |               | 2.6 [DEPENDENT SPOUSE NAME] – Passport |
                |               | 2.7 [DEPENDENT SPOUSE NAME] – Marriage Certificate |
                |               | 2.8 [DEPENDENT SPOUSE NAME] – Birth Certificate |
                |               | 2.9 [CHILD NAME] – Passport |
                |               | 2.10 [CHILD NAME] – Birth Certificate |
                | **Exhibit 3** | **Support Letter** |
                |---------------|--------------------|
                |               | 3.1 [COMPANY NAME] – Letter in Support of the Beneficiary |
                | **Exhibit 4** | **Applicant’s Educational Credentials** |
                |---------------|-----------------------------------------|
                |               | 4.1 Beneficiary’s Diploma |
                |               | 4.2 Beneficiary’s Certifications |
                |               | 4.3 Beneficiary’s Transcript of Records |
                | **Exhibit 5** | **Applicant’s Work Experience** |
                |---------------|---------------------------------|
                |               | 5.1 Curriculum Vitae / Resume |
                |               | 5.2 Certification of Employment |
                |               | 5.3 Trainings |
                | **Exhibit 6** | **Offer Letter with Summary of Terms** |
                |---------------|----------------------------------------|
                |               | 6.1 [COMPANY NAME] – Petitioner’s Job Offer |
                | **Exhibit 7** | **Employer’s Background Information** |
                |---------------|---------------------------------------|
                |               | 7.1 Articles of Incorporation / Organization |
                |               | 7.2 IRS – EIN Confirmation Number |
                |               | 7.3 Company Activities |
                |               | 7.4 Office Photos / Company Website |
                | **Exhibit 8** | **Intent to Depart** |
                |---------------|----------------------|
                |               | 8.1 Beneficiary’s Declaration of Intent to Depart |
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 6.Ensure the tone is professional and concise. Enclose the entire exhibit is within triple backticks like this: ``` Your Exhibit content here ```.               
                """
            ),
        ),
    }

RETRIEVAL_HINTS = {
    "Petition Cover Letter": [
        "employment authorization",
        "Form I-765",
        "Application for Employment Authorization",
        "g-28",
        "g-1145",
        "passport",
        "visa-pages",
        "passport-photo",
        "alien-registration",
        "reentry-permit",
        "EAD",
        "USCIS",
    ],
    "Eligibility Memorandum": [
        "eligibility memorandum",
        "employment authorization",
        "Form I-765",
        "eligibility category",
        "current immigration status",
        "alien registration number",
        "passport",
        "visa-pages",
        "reentry-permit",
        "prior EAD",
        "government-issued photo ID",
        "USCIS",
    ],
    "Evidence-Organization Chart": [
        "evidence organization chart",
        "organization chart",
        "employment authorization",
        "Form I-765",
        "applicant role",
        "employer",
        "department",
        "reporting lines",
        "job duties",
        "supporting evidence",
    ],
    "Exhibit List": [
        "exhibit list",
        "employment authorization",
        "Form I-765",
        "g-28",
        "g-1145",
        "passport-photo",
        "passport",
        "visa-pages",
        "alien-registration",
        "reentry-permit",
        "Form I-94",
        "prior EAD",
        "photo ID",
    ],
}


def build_retrieval_query(file_type: str) -> str:
    hints = RETRIEVAL_HINTS.get(file_type, [])
    return " | ".join([file_type, *hints])


def deduplicate_retrieved_context(retrieved_context) -> list:
    deduplicated = []
    seen_keys = set()

    for document in retrieved_context:
        metadata = getattr(document, "metadata", {}) or {}
        key = (
            str(metadata.get("file_hash", "")),
            str(metadata.get("page_number", "")),
            str(metadata.get("chunk_index", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append(document)

    return deduplicated


def build_retrieved_case_record(retrieved_context) -> str:
    sections = []
    for index, document in enumerate(deduplicate_retrieved_context(retrieved_context), start=1):
        metadata = getattr(document, "metadata", {}) or {}
        sections.append(
            "\n".join(
                [
                    f"### Retrieved Chunk {index}",
                    f"- Source: {metadata.get('source_name', 'unknown')}",
                    f"- Category: {metadata.get('source_category', 'unknown')}",
                    f"- Page: {metadata.get('page_number', 'unknown')}",
                    f"- Chunk: {metadata.get('chunk_index', 'unknown')}",
                    f"- Extraction mode: {metadata.get('extraction_mode', 'unknown')}",
                    document.page_content.strip(),
                ]
            ).strip()
        )

    if not sections:
        return "No retrieved case record was available."

    return "\n\n".join(sections)


def summarise_source_manifest(source_manifest: list[dict]) -> str:
    lines = []
    for entry in source_manifest:
        lines.append(
            "\n".join(
                [
                    f"- Source name: {entry.get('original_filename', 'unknown')}",
                    f"  Category: {entry.get('name', 'unknown')}",
                    f"  File hash: {entry.get('file_hash', 'unknown')}",
                    f"  Extension: {entry.get('extension', 'unknown')}",
                    f"  MIME type: {entry.get('content_type', 'unknown')}",
                    f"  Extraction mode: {entry.get('extraction_mode', 'unknown')}",
                    f"  Pages: {entry.get('page_count', 'unknown')}",
                ]
            )
        )

    return "\n".join(lines) if lines else "- No source manifest available."


def get_document_template(file_type: str) -> str:
    prompt = build_prompt_registry().get(file_type)
    if not prompt:
        raise ValueError(f"No prompt found for document type: {file_type}")
    return prompt.template.strip()


def build_generation_prompt(file_type: str, retrieved_context, source_manifest: list[dict]) -> str:
    prompt_registry = build_prompt_registry()
    prompt = prompt_registry.get(file_type)
    if not prompt:
        raise ValueError(f"No prompt found for document type: {file_type}")

    return "\n\n".join(
        [ 
            prompt.template.strip(),
            "# Retrieved Case Record",
            build_retrieved_case_record(retrieved_context),
            "# Source Manifest",
            summarise_source_manifest(source_manifest),
            "# Additional Output Rules for all the genrated AI Doc",
            "Treat the template as a structural guide only,copy it verbatim and structure.Output must match structure 100% NOTE : We are generating legal document so the below instruction must be put in followed",
            "Act like a lawyer: analyze the retrieved case record and the source manifest, then draft a document grounded in those materials.",
            "Use the retrieved case record as the primary source of factual support and the source manifest as supporting evidence.",
            "When a template field or placeholder is not directly available, look for equivalent or related evidence in the retrieved case record/source manifest and use that to fill the section.",
            "For example, if the template asks for 'petitioner' or 'employer' and the case record uses a different but equivalent term, use the correct party from the evidence.",
            "Fill in every relevant section with facts supported by the retrieved case record or source manifest; if evidence is missing, leave the relevant content blank or mark it as [Not provided] rather than inventing facts.",
            "If key facts are missing, leave the relevant placeholders blank.",
            "Return only the final document enclosed in triple backticks.",
            "# Placeholder Resolution Rules",
            "Every bracketed placeholder and every slash-separated option (Mr. / Ms., he/she/his/her) must be resolved. No bracket, blank, or unresolved '/' option may remain in the final letter.",
            "Fill names, dates, and facts only from the Retrieved Case Record. If a fact is not in the record, write [MISSING: <field name>] instead of leaving the placeholder or inventing a value.",
            "Determine gender from the beneficiary's documents in the Retrieved Case Record and use one consistent form (Mr./Ms., he/she) throughout the letter. If gender cannot be determined, use the beneficiary's full name instead of a pronoun.",
            "If the internal job title does not match a listed Appendix 2 profession, do not silently pick one — output [REVIEW: internal title does not match a listed TN profession].",
            #"Before generating, confirm the beneficiary's citizenship is stated as Canadian or Mexican in the Retrieved Case Record. If citizenship is missing, unclear, or neither Canadian nor Mexican, stop and output only: [STOP: beneficiary citizenship not confirmed as Canadian or Mexican — TN classification requires this].",
            "Duty bullets must use real Markdown bullet syntax, not bold placeholder lines. Populate only as many bullets as the case record supports — do not pad or truncate to reach a fixed number.",
            "Degree and Major must be resolved separately. If no degree is found, use [MISSING: education credential] rather than leaving one field blank.",
            "Before returning output, scan for any remaining '[', ']', or stray '/' characters not part of normal punctuation (e.g., dates). Resolve or tag each with [MISSING: ...] or [REVIEW: ...] before finalizing."
            "# Exhibit Numbering Rule (NOTE: This rule do not apply to the actual exhibit list only file like cover, support letter etc)",
            "Exhibit numbers must be taken ONLY from the (number of Source Manifest) of the file.",
            #"NOTE: Do not use exhibit numbers from any example, template, or prior case use the exhibit you are provided.", 
            #"NOTE: On the exhibit maintain the structure like in the template only the content should change(this means copy the structure not the content)",
            "If a document type isn't in the Source Manifest, write [Exhibit — not provided] instead of guessing a number.",
            "Rules for Exhibit:",
            "Number exhibits sequentially starting from 1.",
            "Only include exhibits that were actually provided.",
            "Keep each description formal, concise, and on its own line.",
            "When multiple forms/fees belong together, put them all under the same Exhibit number (as shown in Style B) and do not give them separate exhibit numbers.",
            "Do not add bold, asterisks, extra spaces, or commentary.",
            "Do not invent any forms, fees, or documents.",
            "# Exhibit Numbering Rule and Format (NOTE: For exhibit only, this rule can over right the rules for all Doc)",
            "Exhibit List Rule:",
            "Strictly preserve the exact structure, numbering, formatting, and layout of the Exhibit List exactly as it appears in the template.Note this more",
            "Only the content (document names, descriptions, amounts, or names) may change.",
            "Do not change the heading style, table format, or the way exhibits are presented.",
            "NOTE:The numbers should be in the file column (left) not in the exhibit number column(right)",
            "No paragraph is allowed in the Exhibit file",
            "It should be similar to the template provided in structure and layout that means treat the template as a structural guide only",
            "# Job Duties Section Rule and Format (NOTE: For the Job Title/Duties section only, this rule can override the general rules for all Docs)",
            "Duties must be pulled only from the beneficiary's Job Description (JD) file — not invented, summarized from unrelated exhibits, or copied from this template's example phrasing.",
            "In the source JD, duties are not always under a section literally named 'Duties' — treat any of the following section headers as valid duty sources: 'Responsibilities', 'Key Responsibilities', 'Job Responsibilities', 'Duties and Responsibilities', 'Key Requirements', 'Requirements', 'Role Overview', 'Core Duties', 'Essential Functions', 'What You'll Do', 'Day-to-Day Activities', or an unlabeled bullet list directly under the job title.",
            "Do not pull bullets from 'Qualifications', 'Requirements' sections that describe candidate skills/education rather than job tasks (e.g., 'Bachelor's degree required' is a qualification, not a duty) — only action-based bullets describing what the person does in the role count as duties.",
            "Populate exactly as many bullets as the source JD supports — do not pad to reach 10 placeholder duties, and do not truncate real duties to fit fewer.",
            "Each duty bullet must be real Markdown bullet syntax (leading '- '), not bold placeholder text like '[Duty 1 in bullet point]'.",
            "If the JD file cannot be found or contains no duty-type content, output '[MISSING: job duties — no JD file found]' instead of generating generic or invented duties.",
            "Treat the numbered '[Duty 1]' through '[Duty 10]' placeholders in the template as a structural guide only (i.e., 'duties go here as a list'), not as a required count or as literal content to preserve.",
            "The job title must be the actual job title only (plain text, no brackets or placeholder formatting). It is not a placeholder and appears above the duties list."
        ]
    ).strip()


async def generate_document(file_type, retrieved_context, source_manifest):
    prompt_text = build_generation_prompt(file_type, retrieved_context, source_manifest)
    client = get_openai_client()
    logger.info("Generating %s with %s retrieved chunks", file_type, len(retrieved_context))

    response = await asyncio.to_thread(
        client.responses.create,
        model=GENERATION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    }
                ],
            }
        ],
    )
    return (response.output_text or "").strip()
