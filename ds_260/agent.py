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
        "DS-260 Completion Guide": DocumentPrompt(
            name="DS-260 Completion Guide Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for DS-260 (Immigrant Visas).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required Forms
                    - Form DS-260, Immigrant Visa Electronic Application Travel.gov
                - Supporting Documents
                    - Must Have
                        - Passport and Civil Documents (birth/marriage certs) Travel.gov
                        - Police Certificates Travel.gov
                        - Medical Exam Report Travel.gov
                        - Affidavit of Support (Form I-864) Travel.gov
                    - Less Demanded
                        - DS-260 Confirmation Page
                        - Supplemental financial documents uploaded to CEAC
                **Step 2**: Use the following structure for the letter:
                ``` 
                ## DS‑260 Completion Guide

                - **Purpose & Access Requirements**  
                Form DS‑260 is the required **Immigrant Visa Electronic Application** for applicants applying for permanent residency at a U.S. consulate abroad. Before starting, applicants must have their **NVC Case Number**, **Beneficiary ID**, and **Invoice ID** from the National Visa Center welcome notice to log in to CEAC (Consular Electronic Application Center). All responses must be entered in **English using Roman alphabet characters**, including transliteration of names or addresses written originally in another alphabet.

                - **Information & Documentation to Gather**  
                Before beginning the DS‑260, applicants should assemble:  
                • Passport bio‑page and personal data;  
                • Full residential history since age 16—including all addresses where the applicant physically resided, even if unofficial or temporary;  
                • Employment and education history;  
                • Family information (spouse, parents, all children, including adopted or stepchildren, regardless of age or whether traveling with the applicant);  
                • Social media usernames, which are required to be disclosed under recent policy updates.

                - **Filling Out & Reviewing the Form**  
                Complete each required field carefully—most fields are mandatory, and the form will not allow submission until all applicable fields are completed. Optional fields may be left blank or marked “Does Not Apply”. Use the **Save** button frequently during data entry, as CEAC times out after approximately 20 minutes of inactivity; saved data persists, but unsaved entries will be lost. Thoroughly review all answers before clicking **Sign and Submit**, since corrections cannot be made afterward; any errors discovered after submission must be addressed directly with the consular officer during the visa interview.

                - **Confirmation Page & Interview Preparation**  
                On successful submission, print the DS‑260 **confirmation page**, which is mandatory to bring to the U.S. consular interview . Do not bring the full DS‑260—it is accessible electronically by the interviewing officer. Prepare to discuss all submitted answers during the interview.

                - **Supporting Civil Documents Submission**  
                After submission, applicants and accompanying family members must collect and upload **civil documents** per NVC instructions. These include but are not limited to birth certificates, passports, marriage or divorce documents, police clearance certificates for any country lived in for six months or more after age 16, and military records if applicable. All documents not in English must have certified translations. Submit as copies, retaining your originals for the interview.

                - **Fees & Timeline**  
                The DS‑260 application processing fee is **US $325**, plus an Affidavit of Support fee (typically US $120) if filing a family‑based immigrant petition; following visa issuance, a USCIS immigrant fee of approximately US $235 is paid to receive the physical green card. NVC document review and processing typically take a few weeks, and interview scheduling depends on embassy availability and visa category.

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
                You are tasked with generating a DS-260 Completion Guide for DS-260 (Immigrant Visas).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required Forms
                    - Form DS-260, Immigrant Visa Electronic Application Travel.gov
                - Supporting Documents
                    - Must Have
                        - Passport and Civil Documents (birth/marriage certs) Travel.gov
                        - Police Certificates Travel.gov
                        - Medical Exam Report Travel.gov
                        - Affidavit of Support (Form I-864) Travel.gov
                    - Less Demanded
                        - DS-260 Confirmation Page
                        - Supplemental financial documents uploaded to CEAC
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
        "Interview Prep Outline": DocumentPrompt(
            name="Interview Prep Outline",
            template=(
                rf"""

                Today’s date is {current_date}.
                You are tasked with generating a Interview Prep Outline for DS-260 (Immigrant Visas).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required Forms
                    - Form DS-260, Immigrant Visa Electronic Application Travel.gov
                - Supporting Documents
                    - Must Have
                        - Passport and Civil Documents (birth/marriage certs) Travel.gov
                        - Police Certificates Travel.gov
                        - Medical Exam Report Travel.gov
                        - Affidavit of Support (Form I-864) Travel.gov
                    - Less Demanded
                        - DS-260 Confirmation Page
                        - Supplemental financial documents uploaded to CEAC
                **Step 2**: Use the following structure for the letter:
                ```
                ## Interview Prep Outline

                ### Documents & Logistics
                Prepare a complete interview folder. Bring your DS‑260 confirmation page, valid passport (with at least six months of validity beyond intended U.S. entry), appointment letter issued by NVC, two identical color photographs meeting U.S. specifications, original or certified civil documents uploaded via CEAC (birth, marriage, police certificates, translations if needed), and sealed medical exam results from an embassy‑approved panel physician. Confirm any embassy‑specific courier instructions in advance. Do not rely on financial or support documents already submitted to NVC unless specifically requested. Keep documents organized and readily accessible to streamline the interview process. 

                ### Dress, Arrival & Conduct
                Dress in professional, conservative attire that reflects respect and seriousness. Arrive at the embassy or consulate no more than 15 minutes early to avoid line complications, and always follow security protocols. Bring snacks or water if long waits are expected. Remain calm, polite, and patient throughout. At the interview window, follow instructions clearly and avoid unnecessary gestures or commentary. First impressions matter—consistent, clear, respectful interaction enhances credibility. 

                ### Review Your DS‑260 & Supporting Info
                Re‑familiarize yourself thoroughly with every detail you entered on DS‑260, including travel history, employment, addresses, and personal relationships. Be prepared to explain any gaps, inconsistencies, or updates. Misalignment between your interview responses and DS‑260 entries may trigger follow‑up or denial. Ensure civil documents match the DS‑260 data and be ready to clarify changes since submission. Accuracy, consistency, and readiness to explain any variation is key to passing the interview. 

                ### Anticipated Interview Questions
                Expect to be asked about:
                - Your purpose for immigrant travel
                - Where you'll live in the U.S.
                - Your affiliations and qualifying relationships
                - Previous U.S. visits and visa history
                - Financial and employment plans
                - Separate marriage‑based cases: meeting details, family backgrounds, shared life
                Answer truthfully, succinctly—only when asked—and avoid volunteering unprompted information. If uncertain, politely ask for clarification rather than guessing.

                ### How to Respond: Style & Attitude
                Speak at a steady pace, in English or your native language if permitted; clarity and honesty are essential. Listen fully before answering; pause briefly to formulate your response. Keep answers concise and focused—consular officers have limited time. Remain calm, courteous, and on‑topic. Avoid over‑explaining or becoming argumentative. If asked for clarification, respond politely. If unsure of a question, say so rather than guess. Maintaining composure underpins credibility. 

                ### After the Interview
                Be prepared to receive a 221(g) request if additional documentation is necessary; ask politely for written instructions and details. If approved, you’ll receive a sealed visa packet and passport; review your visa carefully for errors (name, date of birth, case number). Don’t make permanent housing or job decisions before issuance. If denied on grounds like public charge or incomplete evidence, request specific guidance for correction or waiver eligibility. Above all, follow instructions and keep documentation secure.
 
              
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
        "Document-Upload Memo": DocumentPrompt(
            name="Document-Upload Memo Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Document-Upload Memo for DS-260 (Immigrant Visas).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required Forms
                    - Form DS-260, Immigrant Visa Electronic Application Travel.gov
                - Supporting Documents
                    - Must Have
                        - Passport and Civil Documents (birth/marriage certs) Travel.gov
                        - Police Certificates Travel.gov
                        - Medical Exam Report Travel.gov
                        - Affidavit of Support (Form I-864) Travel.gov
                    - Less Demanded
                        - DS-260 Confirmation Page
                        - Supplemental financial documents uploaded to CEAC
                **Step 2**: Use the following structure for the letter:
                ```
                ## Document‑Upload Memo
    
                **Purpose:** This memo outlines the required procedure for uploading civil and financial documents to the National Visa Center (NVC) via the Consular Electronic Application Center (CEAC) once the DS‑260 Immigrant Visa Application has been submitted.
    
                **Instructions for Uploading Documents (≈ 100 words):**  
                After submitting Form DS‑260, the applicant (or their representative) must log into CEAC using the NVC case number and invoice ID. Navigate to the “Civil Documents” and “Affidavit of Support Documents & Financial Evidence” sections, and click the **“Start Now”** buttons to begin uploading files. Each required document must be uploaded as a separate PDF, JPG, or JPEG file under 2 MB, in color if the original is in color, and named clearly (e.g. "Smith_John_birthcertificate.jpg"). Only after all required uploads are complete will the **“Submit Documents”** button activate; pressing it places the case in queue for NVC review .
    
                **Accepted Document Types & File Requirements (≈ 100 words):**  
                CEAC requires files in JPG, JPEG, or PDF formats, each no larger than 2 MB. Upload one document per file, properly labeled with case number and document type (e.g., “Lastname_Case123_passport.pdf”). Required documents typically include the passport biographical page, birth certificate, marriage certificate (if applicable), police clearance certificates, and any additional civil or financial evidence as outlined in the NVC-generated checklist. Do not send original documents; only clear, legible scans or photos are permitted for electronic submission. Originals must be brought to the consular interview.
    
                **Post‑Upload Workflow (≈ 100 words):**  
                Once uploads are completed and the “Submit Documents” button is clicked, the application moves into NVC’s queue for review. If any document is missing or non‑compliant, NVC will update the status and request additional uploads; applicants can return to CEAC later to correct or add files and must resubmit by clicking “Submit Documents” again. After approving all items, NVC marks the case as “documentarily complete” and coordinates with the U.S. Embassy or Consulate to schedule the visa interview. The applicant should not mail documents unless explicitly instructed by NVC .


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
            You are tasked with generating a RFE Response for DS-260 (Immigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required Forms
                    - Form DS-260, Immigrant Visa Electronic Application Travel.gov
                - Supporting Documents
                    - Must Have
                        - Passport and Civil Documents (birth/marriage certs) Travel.gov
                        - Police Certificates Travel.gov
                        - Medical Exam Report Travel.gov
                        - Affidavit of Support (Form I-864) Travel.gov
                    - Less Demanded
                        - DS-260 Confirmation Page
                        - Supplemental financial documents uploaded to CEAC
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
    "DS-260 Completion Guide": [
        "DS-260",
        "immigrant visa",
        "completion guide",
        "CEAC",
        "NVC",
        "passport",
        "police-certificates",
        "ds-260-confirmation",
        "form-i-693",
        "form-i-864",
        "financial-audits",
        "civil documents",
    ],
    "Exhibit List": [
        "exhibit list",
        "DS-260",
        "immigrant visa",
        "passport",
        "police-certificates",
        "ds-260-confirmation",
        "form-i-693",
        "form-i-864",
        "financial-audits",
        "birth certificate",
        "marriage certificate",
        "civil documents",
        "supporting documents",
    ],
    "Interview Prep Outline": [
        "interview prep outline",
        "DS-260",
        "immigrant visa",
        "consular interview",
        "NVC appointment",
        "passport",
        "police-certificates",
        "ds-260-confirmation",
        "form-i-864",
        "medical exam",
        "civil documents",
    ],
    "Document-Upload Memo": [
        "document upload memo",
        "CEAC upload",
        "DS-260",
        "NVC",
        "passport",
        "police-certificates",
        "ds-260-confirmation",
        "form-i-693",
        "form-i-864",
        "financial-audits",
        "civil documents",
        "certified translations",
    ],
    "RFE Response Brief": [
        "request for evidence response",
        "RFE",
        "221(g)",
        "DS-260",
        "immigrant visa",
        "consular officer",
        "NVC",
        "CEAC",
        "passport",
        "police-certificates",
        "ds-260-confirmation",
        "form-i-864",
        "financial-audits",
        "civil documents",
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
            "Treat the template as a structural guide only; do not copy it verbatim.",
            "Act like a lawyer: analyze the retrieved case record and the source manifest, then draft a document grounded in those materials.",
            "Use the retrieved case record as the primary source of factual support and the source manifest as supporting evidence.",
            "When a template field or placeholder is not directly available, look for equivalent or related evidence in the retrieved case record/source manifest and use that to fill the section.",
            "For example, if the template asks for 'petitioner' or 'employer' and the case record uses a different but equivalent term, use the correct party from the evidence.",
            "Fill in every relevant section with facts supported by the retrieved case record or source manifest; if evidence is missing, leave the relevant content blank or mark it as [Not provided] rather than inventing facts.",
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

