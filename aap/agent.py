import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from .openai_client import get_openai_client


logger = logging.getLogger(__name__)

GENERATION_MODEL = "o3-mini"


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
            name="Petition Cover Letter Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a petition cover letter for an Application for Advance Parole / Travel Authorization (Form I-131).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary.
                - Employer details.
                - Job description and duties.
                - Supporting evidence such as Form I-129, the H-Classification Supplement, certified LCA, and evidence of degree.
                - Required forms:
                  - Form I-131 (Application for Travel Document)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Evidence of pending I-485 or other qualifying application/petition (I-797C receipt notice)
                  - Evidence supporting reason for travel (e.g., medical records, employment letters, school enrollment, family event documents)
                  - Passport (biographic page) and any U.S. visa page
                  - Form I-94 (Arrival/Departure Record)
                  - Prior Advance Parole documents (front and back), if any
                  - Proof of identity (government-issued photo ID)
                  - Proof of current immigration status
                  - Employment verification letter (if applicable)
                  - Proof of relationship to family members abroad (if travel reason is family-based)
                  - Any relevant USCIS notices (I-797)
                  - Two passport-style photos (per USCIS specifications)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Application for Advance Parole / Travel Authorization (Form I-131)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **A-Number:** [Insert A-Number or leave blank]  
                
                Dear Sir/Madam:  
                
                Please accept the enclosed Form I-131, *Application for Travel Document*, on behalf of [Applicant’s Full Name]. This request seeks Advance Parole / Travel Authorization for [briefly state reason for travel, e.g., humanitarian, employment-related, educational, family emergency], in connection with [Applicant’s] [current status or pending application/petition, e.g., pending Form I-485, Adjustment of Status].  
                
                [Applicant’s] current immigration status is [status/class of admission], with an I-94 number of [I-94 Number], most recently admitted to the United States on [Entry Date] at [Port of Entry]. This application is being filed as a(n) [Initial / Renewal] request.  
                
                The following documents are enclosed in support of this application:  
                
                1. Form G-1145, *E-Notification of Application/Petition Acceptance*  
                2. Form G-28, *Notice of Entry of Appearance as Attorney* (if represented)  
                3. Form I-131, *Application for Travel Document*  
                4. Copy of passport biographic page and U.S. visa (if available)  
                5. Form I-94, *Arrival/Departure Record*  
                6. Two passport-style photos (per USCIS specifications)  
                7. Evidence of pending I-485 or other qualifying petition/application (Form I-797C receipt notice)  
                8. Supporting documentation for travel reason (e.g., employer letter, medical documentation, family event evidence)  
                9. Prior Advance Parole document(s) (front and back), if applicable  
                10. Any other relevant USCIS notices (Form I-797)  
                
                We respectfully request favorable adjudication of this application at your earliest convenience. Thank you for your consideration.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_,
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
        "Exhibit List": DocumentPrompt(
            name="Exhibit List Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a petition cover letter for an Application for Advance Parole / Travel Authorization (Form I-131).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary.
                - Employer details.
                - Job description and duties.
                - Supporting evidence such as Form I-129, the H-Classification Supplement, certified LCA, and evidence of degree.
                - Required forms:
                  - Form I-131 (Application for Travel Document)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Evidence of pending I-485 or other qualifying application/petition (I-797C receipt notice)
                  - Evidence supporting reason for travel (e.g., medical records, employment letters, school enrollment, family event documents)
                  - Passport (biographic page) and any U.S. visa page
                  - Form I-94 (Arrival/Departure Record)
                  - Prior Advance Parole documents (front and back), if any
                  - Proof of identity (government-issued photo ID)
                  - Proof of current immigration status
                  - Employment verification letter (if applicable)
                  - Proof of relationship to family members abroad (if travel reason is family-based)
                  - Any relevant USCIS notices (I-797)
                  - Two passport-style photos (per USCIS specifications)

                **Step 2**: Use the following structure for the letter:
                ```
                                                    Exhibit List  
                                    Self-Petitioner: [Beneficiary’s Full Name]  
                                    Position: [Beneficiary’s Position/Title]  

                Exhibit 1:  [Description of Exhibit 1]  
                Exhibit 2:  [Description of Exhibit 2]  
                Exhibit 3:  [Description of Exhibit 3]  
                Exhibit 4:  [Description of Exhibit 4]  
                Exhibit 5:  [Description of Exhibit 5]  
                Exhibit 6:  [Description of Exhibit 6]  
                Exhibit 7:  [Description of Exhibit 7]  
                Exhibit 8:  [Description of Exhibit 8]  
                Exhibit 9:  [Description of Exhibit 9]  
                Exhibit 10: [Description of Exhibit 10]  
                Exhibit 11: [Description of Exhibit 11]  
                Exhibit 12: [Description of Exhibit 12]  
                Exhibit 13: [Description of Exhibit 13]  
                Exhibit 14: [Description of Exhibit 14]  
                Exhibit 15: [Description of Exhibit 15]  
                Exhibit 16: [Description of Exhibit 16]  
                Exhibit 17: [Description of Exhibit 17]  
                Exhibit 18: [Description of Exhibit 18]  
                Exhibit 19: [Description of Exhibit 19]  
                Exhibit 20: [Description of Exhibit 20]  
                Exhibit 21: [Description of Exhibit 21]  
                Exhibit 22: [Description of Exhibit 22]  
                Exhibit 23: [Description of Exhibit 23]  
                Exhibit 24: [Description of Exhibit 24]  
                Exhibit 25: [Description of Exhibit 25]  
                Exhibit 26: [Description of Exhibit 26]  
                Exhibit 27: [Description of Exhibit 27]  
                Exhibit 28: [Description of Exhibit 28]  
                Exhibit 29: [Description of Exhibit 29]  
                Exhibit 30: [Description of Exhibit 30]  
                Exhibit 31: [Description of Exhibit 31]  
                Exhibit 32: [Description of Exhibit 32]  
                Exhibit 33: [Description of Exhibit 33]  
                Exhibit 34: [Description of Exhibit 34]  

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
                You are tasked with generating a petition cover letter for an Application for Advance Parole / Travel Authorization (Form I-131).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary.
                - Employer details.
                - Job description and duties.
                - Supporting evidence such as Form I-129, the H-Classification Supplement, certified LCA, and evidence of degree.
                - Required forms:
                  - Form I-131 (Application for Travel Document)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Evidence of pending I-485 or other qualifying application/petition (I-797C receipt notice)
                  - Evidence supporting reason for travel (e.g., medical records, employment letters, school enrollment, family event documents)
                  - Passport (biographic page) and any U.S. visa page
                  - Form I-94 (Arrival/Departure Record)
                  - Prior Advance Parole documents (front and back), if any
                  - Proof of identity (government-issued photo ID)
                  - Proof of current immigration status
                  - Employment verification letter (if applicable)
                  - Proof of relationship to family members abroad (if travel reason is family-based)
                  - Any relevant USCIS notices (I-797)
                  - Two passport-style photos (per USCIS specifications)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Eligibility Memorandum in Support of Application for Advance Parole / Travel Authorization (Form I-131)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **A-Number:** [Insert A-Number or leave blank]  
                
                Dear Sir/Madam:  
                
                This memorandum is submitted in support of [Applicant’s Full Name]’s Form I-131, *Application for Travel Document*, seeking Advance Parole / Travel Authorization. The purpose of this memorandum is to outline the applicant’s immigration background, establish eligibility for Advance Parole under applicable regulations, and summarize the evidence provided.  
                
                **Background**  
                [Applicant’s Full Name] is currently in [status/class of admission], with an I-94 number of [I-94 Number], last admitted to the United States on [Entry Date] at [Port of Entry]. The applicant has a [pending Form I-485 / other qualifying application or petition] filed with USCIS under receipt number [Receipt Number], which serves as the basis for eligibility.  
                
                **Eligibility Basis**  
                Pursuant to 8 CFR 223, the applicant is eligible to request Advance Parole as [brief statement of eligibility — e.g., an applicant for adjustment of status under INA §245]. The requested travel authorization is sought for [reason for travel, e.g., humanitarian, employment, education, family-related] and is supported by the enclosed documentation.  
                
                **Enclosures in Support of This Memorandum**  
                1. Form I-131, *Application for Travel Document*  
                2. Form G-1145, *E-Notification of Application/Petition Acceptance*  
                3. Form G-28, *Notice of Entry of Appearance as Attorney* (if represented)  
                4. Copy of passport biographic page and U.S. visa (if available)  
                5. Form I-94, *Arrival/Departure Record*  
                6. Two passport-style photos (per USCIS specifications)  
                7. Evidence of pending I-485 or other qualifying petition/application (Form I-797C receipt notice)  
                8. Supporting documentation for reason for travel (e.g., employer letter, medical documentation, family event evidence)  
                9. Prior Advance Parole document(s) (front and back), if applicable  
                10. Any other relevant USCIS notices (Form I-797)  
                
                **Conclusion**  
                Based on the information provided and the supporting evidence submitted, [Applicant’s Full Name] meets the eligibility requirements for Advance Parole / Travel Authorization. We respectfully request favorable adjudication of this application.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_,
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
                You are tasked with generating a petition cover letter for an Application for Advance Parole / Travel Authorization (Form I-131).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary.
                - Employer details.
                - Job description and duties.
                - Supporting evidence such as Form I-129, the H-Classification Supplement, certified LCA, and evidence of degree.
                - Required forms:
                  - Form I-131 (Application for Travel Document)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Evidence of pending I-485 or other qualifying application/petition (I-797C receipt notice)
                  - Evidence supporting reason for travel (e.g., medical records, employment letters, school enrollment, family event documents)
                  - Passport (biographic page) and any U.S. visa page
                  - Form I-94 (Arrival/Departure Record)
                  - Prior Advance Parole documents (front and back), if any
                  - Proof of identity (government-issued photo ID)
                  - Proof of current immigration status
                  - Employment verification letter (if applicable)
                  - Proof of relationship to family members abroad (if travel reason is family-based)
                  - Any relevant USCIS notices (I-797)
                  - Two passport-style photos (per USCIS specifications)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Evidence – Organization Chart in Support of Application for Advance Parole / Travel Authorization (Form I-131)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **A-Number:** [Insert A-Number or leave blank]  
                
                Dear Sir/Madam:  
                
                Please find enclosed the organizational chart of [Company/Organization Name] in support of [Applicant’s Full Name]’s Form I-131, *Application for Travel Document*. This evidence is submitted to demonstrate the applicant’s position within the company, the reporting hierarchy, and the connection between the applicant’s role and the need for travel authorization.  
                
                **Purpose of Submission**  
                The organizational chart illustrates:  
                - The applicant’s current or proposed role within the company.  
                - Reporting lines and supervisory relationships.  
                - Departmental structure and the applicant’s relation to other personnel.  
                - How the applicant’s position and duties support the basis for the requested Advance Parole.  
                
                **Enclosures in Support of This Evidence**  
                1. Company organizational chart identifying the applicant’s position.  
                2. Legend or key explaining titles, reporting lines, and departments (if not clearly labeled).  
                3. Supporting documentation (if available in the provided file set), such as:  
                   - Job description corresponding to the position shown.  
                   - Employer letter confirming the role, duties, and travel need.  
                   - Relevant USCIS notices or employment verification records.  
                
                We respectfully submit this organizational chart and related evidence to assist USCIS in understanding [Applicant’s Full Name]’s position within [Company/Organization Name] and its relevance to the Advance Parole request.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_,
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
    }

RETRIEVAL_HINTS = {
    "Petition Cover Letter": [
        "advance parole",
        "travel authorization",
        "Form I-131",
        "Application for Travel Document",
        "form-g-28",
        "form-g-1145",
        "form-i-94",
        "form-i-797",
        "passport",
        "visa page",
        "employer-letter",
        "professional-certs",
        "USCIS",
    ],
    "Exhibit List": [
        "exhibit list",
        "advance parole",
        "travel authorization",
        "form-i-129",
        "form-g-28",
        "form-g-1145",
        "form-i-907",
        "certified-lca",
        "degree-evidence",
        "employer-letter",
        "form-i-94",
        "form-w2",
        "form-i-797",
        "form-w2-1099",
        "professional-certs",
        "passport",
    ],
    "Eligibility Memorandum": [
        "eligibility memorandum",
        "advance parole",
        "Form I-131",
        "pending I-485",
        "qualifying application",
        "travel reason",
        "humanitarian",
        "employment",
        "education",
        "family emergency",
        "form-i-94",
        "form-i-797",
        "passport",
        "visa page",
        "current immigration status",
    ],
    "Evidence-Organization Chart": [
        "evidence organization chart",
        "organization chart",
        "advance parole",
        "travel authorization",
        "employer-letter",
        "job description",
        "department",
        "reporting lines",
        "employment verification",
        "form-i-797",
        "USCIS notice",
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
            "# Additional Output Rules",
            "Use only the retrieved case record and the source manifest.",
            "If key facts are missing, leave the relevant placeholders blank.",
            "Return only the final document enclosed in triple backticks.",
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
