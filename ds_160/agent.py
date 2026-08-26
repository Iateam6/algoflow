import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from case_jobs.integrations.openai_client import get_openai_client
from case_jobs.pipeline.prompt_contract import build_legal_drafting_contract, prepare_template_for_generation


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
        "DS-160 Completion Guide": DocumentPrompt(
            name="DS-160 Completion Guide Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a DS-160 Interview Cheat-Sheet for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page

                **Step 2**: Use the following structure for the letter:
                ``` 
                ## DS‑160 Completion Guide

                **Passport Details**  
                Ensure the applicant provides a valid passport biographical page, bearing full name, date of birth, passport number, country of issuance, and expiration date. The passport must remain valid for at least six months beyond the intended U.S. stay (unless exempt by bilateral agreement). Accurate transcription into the DS‑160 is critical to avoid matching errors during reciprocity or entry. If any passport fields are missing in the source file (e.g. issue date or controlling authority), leave those entries blank rather than guessing.

                **Travel Itinerary**  
                If travel dates, intended U.S. departure and return plans, cities, or purpose of visit are provided, summarize them clearly. Include flight details if available. If no itinerary was included in the DS‑160 file, leave this section blank—do not assume destination or duration. Accurate itinerary detail helps demonstrate intent of temporary stay and supports the consular understanding of trip planning.

                **Previous U.S. Travel History**  
                List any prior visits to the U.S., including dates of entry and exit, visa types used, or prior refusals or overstays if documented. This section helps consular officers assess patterns of compliance or risk. If the DS‑160-derived file lacks travel history information, omit this section entirely to avoid imprinting unverified values and triggering inconsistencies.

                **Education and Employment History**  
                Provide details of the applicant’s education and employment over the past few years as captured in DS‑160: institutions attended, degrees earned, current and former employers, job titles, and dates. These details support evaluation of eligibility for visa categories like F, J, M, work, or business types. If fields are omitted in the source, leave them blank.

                **Social Media Usernames**  
                As of mid‑2025, applicants—including those applying for F, M, and J visas—must list **all** usernames or handles used on social media platforms in the past five years (e.g. Facebook, Instagram, LinkedIn, YouTube, X, Reddit, Tumblr, etc.). If applicants used platforms in the past five years, these identifiers must be entered even if accounts are inactive or deleted. Set these accounts to public during application review if required. If no platforms were used, the applicant may enter "None."

                **Digital Photo Requirements**  
                Confirm that a compliant photo file was uploaded with the DS‑160. The photo must meet U.S. visa specs: white background, 2×2 inches, head size between 50–69% of frame, neutral expression. If the original file lacks photo metadata, leave this placeholder blank and note that a valid digital photo must still be provided before submission.

                **DS‑160 Confirmation Page**  
                Ensure extraction of the confirmation barcode page details: application ID number, barcode, and date of submission. This page is required during the visa interview. If missing in the provided DS‑160 data, leave it blank, but highlight that the confirmation page must be printed and brought to the consular appointment.


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
            name="Exhibit List",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page

                **Step 2**: Use the following structure for the letter:
                ``` 
                **List of Supporting Documents**  
                **DS-260 Visa Application**

                **Petitioner:** [COMPANY NAME]  
                **Beneficiary:** [BENEFICIARY’S NAME]

                Pursuant to the United States-Mexico-Canada Agreement, a foreign national is entitled to enter the United States under the DS-260 Status Visa category. Below is a complete list of supporting documents submitted to establish that **Mr. / Ms. [BENEFICIARY’S NAME]**, a citizen of **[Canada / Mexico]**, is qualified for the DS-260 Visa.

                | **Exhibit 1** | **Forms & Fees** |
                |---------------|------------------|
                |               | 1.1 Cover Letter |
                |               | 1.2 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.3 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.4 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.5 Form G-28 “Notice of Entry of Appearance as Attorney” |
                |               | 1.6 Form I-907 “Request for Premium Processing” |
                |               | 1.7 Form I-129 + DS-260 Supplement “Petition for a Nonimmigrant Worker” |
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
        "Interview Cheat-Sheet": DocumentPrompt(
            name="Interview Cheat-Sheet",
            template=(
                rf"""

                Today’s date is {current_date}.
                You are tasked with generating a  Interview Cheat-Sheet for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page


                **Step 2**: Use the following structure for the letter:
                ```
                ## Interview Cheat‑Sheet

                ### Applicant Identity & Contact Details  
                **Full Name:** [Applicant’s Full Name]  
                **Date of Birth:** [MM/DD/YYYY]  
                **Passport Number:** [Passport Number]  

                The consular officer will first verify your identity and contact information exactly as they appear on your DS‑160 form. Be prepared to state your full name, date of birth, and passport number clearly. Speak confidently and ensure you match the exact spelling and format on official documents. If your DS‑160 included any alias or previous name variations, be ready to explain briefly why you used them and confirm current legal name usage.

                ### Purpose of Trip & Planned Travel Dates  
                **Visa Category:** [e.g., B‑2 Tourist, F‑1 Student]  
                **Purpose of Visit:** [Purpose (from DS‑160)]  
                **Planned Dates of Travel:** [Arrival date] to [Departure date]  

                This section captures your stated reason for visiting the U.S. and the timeframe you expect to stay. Explain why your trip is necessary and appropriate under the specific visa category—noting, for example, tourism, business meetings, academic study, or exchange programs. If dates are approximate, clarify the rationale, such as academic term start or business schedule. Be prepared to justify why you chose that period and how it aligns with your purpose without intention of immigrant intent.

                ### U.S. Contact & Accommodation  
                **U.S. Point of Contact or Hosting Address:** [Name / Organization / Address]  
                **Accommodation Plan:** [Where you will stay]  

                If your DS‑160 provided a U.S. point of contact or address where you will reside, summarize that information. Explain the nature of your relationship with the point of contact or how you arranged your accommodation (hotel, family, host organization). Be ready to articulate why staying there is appropriate for the trip and how it supports the legitimacy of your travel purpose. Avoid speculation—state only what’s filled in your application and ensure consistency.

                ### Employment or Study Details  
                **Current Occupation or Institution:** [Employer or School Name]  
                **Job Title or Course of Study:** [Your position or program]  
                **Duration in Role or Program:** [Start Date – Present]  

                When asked, describe your current professional or academic status as provided in your DS‑160 form. Clarify your role, duties, or field of study and how long you’ve held that role or been enrolled. If applying under a student or exchange category, be sure to reference your SEVIS‑ID and educational institution details. For work or business‑based applications, explain how your trip’s activities relate to your current position or employer responsibilities.

                ### Travel History & Prior U.S. Visits  
                **Previous U.S. Visits (last 5 years):** [List dates & purpose]  
                **International Travel History:** [Other countries visited]  

                You will be asked about your recent travel history—especially to the U.S.—as entered in the DS‑160. Provide details of any past visits, including dates and legitimate reasons (tourism, work, study). If your DS‑160 recorded other international travel, be prepared to describe those trips and demonstrate credible reasons for repeated travel abroad. This shows consistency and helps eliminate concerns of immigration intent. Only speak to what’s filled in your application.

                ### Social Media & Online Presence  
                **Social Media Accounts (active within last 5 years):** [Handles or Links]  

                Modern DS‑160 versions require listing social media usernames active over the past five years. Expect to discuss these accounts if listed—why you used them and how you maintain transparency. Be ready to confirm that you included all accounts and that your online presence corresponds with the identity and background given in the DS‑160. Honesty is critical, and the consular official may check consistency between accounts and your stated personal history.

                ### Funding & Financial Ties  
                **Source of Funds for Trip:** [Personal savings / Sponsor]  
                **Employment Income or Funding Source:** [Details]  
                **Ties to Home Country:** [Assets, family, employment]  

                You may be prompted to explain how you will finance the trip and why you will return. Describe your financial means—such as salary, savings, institutional sponsorship, or family support—as indicated in your DS‑160 responses. Also emphasize ties that strongly root you in your home country, such as stable employment, family obligations, or property ownership. This underscores your intent to depart the U.S. at the end of the authorized stay.

                ### Supporting Document Prep  
                **Documents to Present:** DS‑160 confirmation page, passport, photo, receipt, supporting evidence as indicated.  

                Be ready to show your DS‑160 confirmation barcode page at the interview, along with passport, the photo upload and visa fee receipt. Gather any additional documents you listed in your DS‑160, such as itinerary, employment letters, invitation letters, educational certificates, or any other support. Organize them logically so you can refer quickly if asked. Your preparation will reflect attention to detail and reinforce credibility with the officer.

              
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
        "RFE Response Brief": DocumentPrompt(
            name="RFE Response Brief",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a RFE Response for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page


                **Step 2**: Use the following structure for the letter:
                ```
                # Response to Consular RFE – DS-160 Application for [Applicant’s Full Name]  
                **[Applicant’s Name or Representative's Name]**  
                **[Applicant’s Address or Attorney’s Office]**  
                **[City, State, ZIP Code]**

                **Date:** _YYYY-MM-DD_

                **Subject:** Response to Request for Evidence (RFE) – DS-160 Nonimmigrant Visa Application for [Applicant’s Full Name]

                **Dear Consular Officer,**

                **Introduction & RFE Reference**  
                **Parties & Purpose:** “[Applicant’s Name] (the “Applicant”) respectfully submits this Response to the Request for Evidence (RFE) issued in connection with the DS-160 Nonimmigrant Visa application.”  
                **RFE Details:**  
                Case Number: _[Number]_  
                RFE Issued: _[Date]_  
                Response Deadline: _[Date]_  
                **Summary of Consular Concerns:**  
                1. Discrepancy or lack of clarity in answers provided on DS-160 form.  
                2. Incomplete supporting documentation regarding identity and travel plans.  
                3. Request for further clarification on applicant’s intent and ties to home country.

                **Rebuttal to Consular Concerns**  
                **Concern 1: DS-160 Content Accuracy**  
                **Consular Position:** “Inconsistencies or vague responses in DS-160 responses.”  
                **Rebuttal:**  
                The updated DS-160 form (Ex. A) corrects and clarifies previous responses. A summary sheet outlining key changes and reasons for edits has also been included (Ex. B). The applicant affirms under oath that all details now accurately reflect their circumstances and travel intent.  

                **Concern 2: Supporting Documentation Deficiency**  
                **Consular Position:** “Insufficient documentation provided to verify identity or purpose of travel.”  
                **Rebuttal:**  
                Applicant has now submitted all required materials, including a valid passport (Ex. C), compliant photo (Ex. D), travel itinerary (Ex. E), and DS-160 confirmation page (Ex. F). Where applicable, the resume/CV (Ex. G) and any relevant employment or invitation letters have also been enclosed.  

                **Concern 3: Nonimmigrant Intent Clarification**  
                **Consular Position:** “Need for further evidence of applicant’s intent to return to home country.”  
                **Rebuttal:**  
                Applicant has provided evidence of strong ties to the home country, including family connections, property records, or ongoing educational/professional commitments (Ex. H). These demonstrate a clear intent to return after the temporary visit as stated in the application.

                **Additional Legal Authority:**  
                In accordance with the U.S. Department of State’s regulations regarding nonimmigrant visa issuance, the applicant respectfully affirms full compliance with all DS-160 requirements and guidelines as outlined at Travel.gov.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the revised application materials and supporting documents, the Applicant has fully addressed the consular concerns and remains eligible for issuance of a nonimmigrant visa.”  
                **Request for Adjudication:** “Applicant respectfully requests that the consular officer proceed with adjudication and notify the Applicant at **[Email Address]** of the visa decision or next steps.”  
                **Point of Contact:** “For further questions or submission of additional documents, please contact **[Attorney or Applicant’s Representative Name]**, **[Title, if applicable]**, at **[Phone Number]** or **[Email Address]**.”

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Representative’s Name], [Title if applicable]**  
                **[Organization Name or Applicant]**
 

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
    "DS-160 Completion Guide": [
        "DS-160",
        "nonimmigrant visa",
        "completion guide",
        "passport",
        "passport number",
        "travel history",
        "visa application",
        "police-certificates",
        "ds-260-confirmation",
        "form-i-693",
        "form-i-864",
        "financial-audits",
        "consular",
    ],
    "Exhibit List": [
        "exhibit list",
        "DS-160",
        "nonimmigrant visa",
        "passport",
        "photo",
        "travel itinerary",
        "employment letter",
        "invitation letter",
        "financial-audits",
        "supporting documents",
    ],
    "Interview Cheat-Sheet": [
        "interview cheat sheet",
        "DS-160",
        "consular interview",
        "nonimmigrant visa",
        "passport",
        "travel purpose",
        "nonimmigrant intent",
        "ties to home country",
        "employment",
        "family",
        "financial support",
    ],
    "RFE Response Brief": [
        "request for evidence response",
        "RFE",
        "221(g)",
        "DS-160",
        "nonimmigrant visa",
        "consular officer",
        "passport",
        "photo",
        "travel itinerary",
        "employment letter",
        "invitation letter",
        "ties to home country",
        "supporting documents",
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
    print("\n" + "=" * 100)
    print("RETRIEVED CASE RECORD DEBUG INFO")
    print("=" * 100)
    print(f"Total chunks included: {len(retrieved_context)}\n")
    
    sections = []
    for index, document in enumerate(retrieved_context, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        chunk_text = document.page_content.strip()
        chunk_section = "\n".join(
            [
                f"### Retrieved Chunk {index}",
                f"- File: {metadata.get('source_name', 'unknown')}",
                chunk_text,
            ]
        ).strip()
        sections.append(chunk_section)
        
        # Print chunk info
        print(f"[Chunk {index}]")
        print(f"  File: {metadata.get('source_name', 'unknown')}")
        print(f"  Content length: {len(chunk_text)} characters")
        print(f"  Content preview (first 200 chars): {chunk_text[:200]}...")
        print(f"  Full content:\n{chunk_text}")
        print("-" * 100)
        
        # Log each chunk
        logger.info("Chunk %d | File: %s | Length: %d chars",
                     index,
                     metadata.get('source_name', 'unknown'),
                     len(chunk_text))

    if not sections:
        print("No retrieved case record was available.")
        print("=" * 100 + "\n")
        return "No retrieved case record was available."

    result = "\n\n".join(sections)
    print(f"Total Retrieved Case Record length: {len(result)} characters")
    print("=" * 100 + "\n")
    return result


def summarise_source_manifest(source_manifest: list[dict]) -> str:
    print("\n" + "=" * 100)
    print("SOURCE MANIFEST DEBUG INFO")
    print("=" * 100)
    print(f"Total files in manifest: {len(source_manifest)}")
    
    lines = []
    for idx, entry in enumerate(source_manifest, start=1):
        manifest_entry = "\n".join(
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
        lines.append(manifest_entry)
        print(f"\nFile {idx}:")
        print(manifest_entry)
    
    print("=" * 100 + "\n")
    result = "\n".join(lines) if lines else "- No source manifest available."
    logger.info("Source Manifest Summary:\n%s", result)
    return result


def get_document_template(file_type: str) -> str:
    prompt = build_prompt_registry().get(file_type)
    if not prompt:
        raise ValueError(f"No prompt found for document type: {file_type}")
    return prompt.template.strip()


def build_generation_prompt(file_type: str, retrieved_context, source_manifest: list[dict]) -> str:
    print("\n" + "=" * 100)
    print("BUILDING GENERATION PROMPT")
    print("=" * 100)
    print(f"Document Type: {file_type}")
    print(f"Total retrieved context chunks: {len(retrieved_context)}")
    print("=" * 100 + "\n")
    
    logger.info("Building generation prompt for: %s", file_type)
    logger.info("Total retrieved context chunks: %d", len(retrieved_context))
    
    prompt_registry = build_prompt_registry()
    prompt = prompt_registry.get(file_type)
    if not prompt:
        raise ValueError(f"No prompt found for document type: {file_type}")

    # Build each section with debugging
    print("[1/5] Building template section...")
    template_section = prepare_template_for_generation(prompt.template)
    print(f"✓ Template length: {len(template_section)} characters\n")
    
    print("[2/5] Building retrieved case record section...")
    case_record_section = build_retrieved_case_record(retrieved_context)
    print(f"✓ Case record length: {len(case_record_section)} characters\n")
    
    print("[3/5] Building legal drafting contract section...")
    legal_contract = build_legal_drafting_contract()
    print(f"✓ Legal contract length: {len(legal_contract)} characters\n")
    
    print("[4/5] Building output rules section...")
    output_rules = "\n".join([
        "# Additional Output Rules for all the genrated AI Doc",
        "Do not provide ideas on how to write the document; only generate the document itself.",
        "Critical rule accuracy: Do not invent any facts or details more so in the exhibit and the context should short just like the template.",
        "Common section headings and subheadings are provided in the template; do not invent new headings or subheadings (should remain the same should not change) so stop adding Employer Address or Beneficiary Address and this sentence **Enclosed, please find the following materials in support of this application:** should be constant .",
        "If the EB-1A profession category use field of study intead of job title and if the job title is not listed in the Appendix 2, use the field of study instead of job title.",
        "Treat the template as a structural guide only; follow its structure and paragraph flow, but replace all example content with details specific to the current visa application. Match the template's approximate length and word count per section — do not condense, summarize, or expand paragraphs beyond what the template's structure implies. The output should read as a complete, single-page letter, mirroring the template's proportions (e.g., a 3-sentence duties description stays 2-3 sentences, not 1 or 5; a 4-paragraph body stays 4 paragraphs, not 2 or 6).",
        "Act like a lawyer: analyze the complete retrieved case record, then draft a document grounded in those materials.",
        "Act like a lawyer: analyze the complete retrieved case record, then draft a document grounded in those materials.",
        "Use the retrieved case record as the source of factual support.",
        "When a template field or placeholder is not directly available, look for equivalent or related evidence in the retrieved case record and use that to fill the section.",
        "For example, if the template asks for 'petitioner' or 'employer' and the case record uses a different but equivalent term, use the correct party from the evidence.",
        "Fill every relevant section only with supported facts; use [MISSING: field name] when required evidence is unavailable.",
        "Never leave a required field blank and never invent a replacement value.",
        "Return only the final document enclosed in triple backticks.",
        "# Placeholder Resolution Rules",
        "Every bracketed placeholder and every slash-separated option (Mr. / Ms., he/she/his/her) must be resolved. No bracket, blank, or unresolved '/' option may remain in the final letter.",
        "Fill names, dates, and facts only from the Retrieved Case Record. If a fact is not in the record, write [MISSING: <field name>] instead of leaving the placeholder or inventing a value.",
        "Determine gender from the beneficiary's documents in the Retrieved Case Record and use one consistent form (Mr./Ms., he/she) throughout the letter. If gender cannot be determined, use the beneficiary's full name instead of a pronoun.",
        # "If the internal job title does not match a listed Appendix 2 profession, do not silently pick one — output [REVIEW: internal title does not match a listed TN profession].",
        "Duty bullets must use real Markdown bullet syntax, not bold placeholder lines. Populate only as many bullets as the case record supports — do not pad or truncate to reach a fixed number.",
        # "Degree and Major must be resolved separately. If no degree is found, use [MISSING: education credential] rather than leaving one field blank.",
        # "Before returning output, scan for any remaining '[', ']', or stray '/' characters not part of normal punctuation (e.g., dates). Resolve or tag each with [MISSING: ...] or [REVIEW: ...] before finalizing."
        "# Exhibit Numbering Rule (NOTE: This rule do not apply to the actual exhibit list only cover letter)",
        "Do not change the order or the number of the exhibits you are provided when generating the exhibit list section in the cover leter ,if its Exhibit 1 to Exhibit 4 do not make to 5 or 7, but use it to reference the exhibits.",
        "Use the provided exhibit list example to show you the tone and style/format to generate the exhibit list SECTION in the cover letter (NOTE: Do not change the order of exhibit you are providing).",
        "Do not use `- or ·` hyphenate, bullet or number the exhibit list.",
        "#Exhibit List Rule:",
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
    ])
    print(f"✓ Output rules length: {len(output_rules)} characters\n")
    
    print("[5/5] Combining all sections...")
    all_sections = [
        template_section,
        "# Retrieved Case Record",
        case_record_section,
        legal_contract,
        output_rules,
    ]
    
    final_prompt = "\n\n".join(all_sections)
    
    print("=" * 100)
    print("FINAL PROMPT READY")
    print("=" * 100)
    print(f"✓ Total final prompt length: {len(final_prompt)} characters")
    print(f"✓ Total sections: {len(all_sections)}")
    print("=" * 100)
    print("\n*** FULL FINAL PROMPT BEING SENT TO OPENAI: ***\n")
    print(final_prompt)
    print("\n" + "=" * 100)
    print("*** END OF FINAL PROMPT ***")
    print("=" * 100 + "\n")
    
    logger.info("Final prompt length: %d characters", len(final_prompt))
    
    return final_prompt.strip()


async def generate_document(file_type, retrieved_context, source_manifest):
    prompt_text = build_generation_prompt(file_type, retrieved_context, source_manifest)
    client = get_openai_client()
    logger.info("Generating %s with %s retrieved chunks", file_type, len(retrieved_context))
    
    # Log the full prompt being sent to the model
    logger.info("=" * 100)
    logger.info("PROMPT BEING SENT TO GENERATION MODEL FOR: %s", file_type)
    logger.info("=" * 100)
    logger.info("PROMPT LENGTH: %d characters", len(prompt_text))
    logger.info("PROMPT CONTENT:\n%s", prompt_text)
    logger.info("=" * 100)
    
    # Also print to console for immediate visibility
    print("\n" + "=" * 100)
    print(f"PROMPT BEING SENT TO GENERATION MODEL FOR: {file_type}")
    print("=" * 100)
    print(f"PROMPT LENGTH: {len(prompt_text)} characters")
    print("PROMPT CONTENT:")
    print(prompt_text)
    print("=" * 100 + "\n")

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
