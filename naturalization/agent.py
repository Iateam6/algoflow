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
        "Petition Cover Letter": DocumentPrompt(
            name="Petition Cover Letter",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a petition cover letter for Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ``` 
                **RE: Application for Reentry Permit (Form I-131)**  
                **Applicant:** [Insert Full Name]  
                **Alien Number:** [Insert A-Number]  
                
                Dear Sir/Madam,
                
                We are submitting the following documents in support of the above applicant’s **Application for Reentry Permit (Form I-131)**:
                
                1 Check in the amount of $[Insert Amount] to cover the **Form I-131** filing fee and **biometric services**  
                2 **Form G-1145**, *e-Notification of Application/Petition Acceptance*  
                3 **Form G-28**, *Notice of Entry of Appearance as Attorney or Accredited Representative*  
                4 **Form I-131**, *Application for Travel Document*  
                5 Copy of the **front and back** of the applicant’s **green card**  
                6 Copy of the applicant’s **passport biographic page**  
                
                We appreciate your prompt attention to this matter.


                Very truly yours,

                \_\_\_\_\_\_\_\_\_\_\_,  
                [Authorized Signatory’s Name]
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
                You are tasked with generating a Exhibit List for Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
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
        "Eligibility-Checklist Memo": DocumentPrompt(
            name="Eligibility-Checklist Memo Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating an Eligibility-Checklist Memo for Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ```
                # Eligibility‑Checklist Memo – Naturalization (Form N‑400)  
                **[Applicant’s Full Name]**  
                **[Applicant’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** Naturalization Eligibility Checklist Memo – [Applicant’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Purpose:** “[Applicant’s Full Name] submits this Eligibility‑Checklist Memo to accompany their Form N‑400, Application for Naturalization, summarizing key eligibility requirements and supporting evidence.”  
                **Basis of Filing:** “Applicant files under [General Provision: 5‑year permanent resident; Married to U.S. Citizen: 3‑year rule; Military service; VAWA, etc.] per INA § [section].”  

                **Eligibility Checklist**  
                Below is a narrative summary confirming that the Applicant meets each naturalization requirement under INA, supported by the evidence in the filing package:

                - **18 years of age or older**: Applicant is **[age]**, born on **[DOB]** (≥ 18 years at filing).  
                - **Lawful Permanent Resident status**: Green Card issued on **[Date]**, Alien Registration Number: **[A‑Number]**.  
                - **Continuous residence requirement**: 
                - Under general eligibility: permanent resident for at least 5 years (residency began on **[Date]**).
                - If eligible via marriage: married and living with U.S. citizen spouse since **[Date]**, meeting the 3‑year rule.  
                - **Physical presence requirement**: Applicant has been physically present in the U.S. for ≥ 30 months during the past 5 years (or ≥ 18 months in past 3 if under 3-year rule), as documented in travel history worksheets.  
                - **Residency in USCIS district / state**: Applicant has lived in **[State/District]** continuously since **[Date]**, satisfying the 3‑month residency requirement.  
                - **English language ability and civics knowledge**: Applicant can read, write, and speak basic English and demonstrates knowledge of U.S. history and government, supported by eligibility under standard testing or applicable exemptions.  
                - **Good moral character**: Applicant has no disqualifying criminal history; any arrests or convictions are detailed with documentation, and eligibility is preserved under relevant INA provisions.  
                - **Attachment to the Constitution**: Applicant affirms willingness to support the Constitution of the United States and take the Oath of Allegiance.  

                **Supporting Evidence**  
                The application packet includes the following supporting documentation:  
                - Copy of both sides of the Permanent Resident Card (Form I‑551)  
                - Travel history record summarizing trips outside the U.S.  
                - Residency and employment history covering the required period  
                - English and civics test results or applicable test waiver documentation (e.g. Form N‑648)  
                - Arrest records, court dispositions, or explanations regarding any criminal record (if applicable)  
                - Signed declaration affirming attachment to the Constitution and willingness to take the Oath  

                **Conclusion & Request**  
                **Eligibility Confirmed:** “Based on the above facts and enclosed evidence, the Applicant clearly satisfies all statutory and regulatory requirements for naturalization under INA eligibility criteria.”  
                **Request for Adjudication:** “Applicant respectfully requests that USCIS accept and adjudicate their N‑400 application and schedule a naturalization interview and oath ceremony without delay.”  

                **Point of Contact:** For any questions or additional information, please contact **[Preparer or Representative Name]** at **[Phone Number]** or **[Email Address]**.  

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Preparer’s Name], [Title or Role]**  
                **[Organization, if applicable]**  
  
              
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
        "Good Moral Character Brief": DocumentPrompt(
            name="Good Moral Character Brief Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating an Good Moral Character Brief for a Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ```
                # Good Moral Character Brief – [Applicant’s Full Name]  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** Good Moral Character Brief – Naturalization of **[Applicant’s Full Name]**  

                **Dear USCIS Officer,**  

                **Background & Statutory Requirement:**  
                Pursuant to INA § 316(e) and 8 CFR § 316.10, the Applicant must demonstrate good moral character (GMC) during the statutory period: the five years immediately preceding filing (or three years if filing based on marriage to a U.S. citizen), and continuing up to the date of the Oath of Allegiance.

                **Evidence of Positive Conduct:**  
                During the required period, the Applicant has maintained steady lawful permanent residence, paid all required federal and state taxes, and complied with community and civic responsibilities. There are no arrests, convictions, or citations during the GMC period or otherwise. The Applicant has demonstrated honesty in all immigration filings and interviews, including full and truthful responses to all questions on Form N‑400 and during prior immigration proceedings.

                **Relevant Supporting Documentation:**  
                - IRS tax transcripts and W‑2 statements for the GMC period confirming timely filing and payment of taxes  
                - Personal affidavits and letters of recommendation attesting to honesty, reliability, and contributions to the local community (e.g. volunteer service, family support)  
                - Evidence of Selective Service registration (if applicable), and no incidents of failure to register during the eligible period  

                **No Disqualifying Conduct:**  
                The Applicant has not committed any crimes involving moral turpitude (CIMTs), aggravated felonies, or unlawful acts that would automatically disqualify GMC under INA §101(f) or 8 CFR § 316.10(b). There have been no false claims to U.S. citizenship, no immigration fraud, and no misuse of public benefits that would reflect dishonesty or poor character.

                **Totality of Circumstances & Character Rehabilitation (if applicable):**  
                There is no adverse conduct outside the statutory period that bears negatively on the Applicant’s present moral character. Even if past incidents existed, they would be outside the GMC period, and the Applicant’s subsequent behavior demonstrates full reformation and alignment with U.S. standards of good moral character.

                **Conclusion & Request:**  
                Based on the foregoing and the attached documentation, the Applicant clearly meets and continues to meet the statutory and regulatory requirements for **Good Moral Character**. The Applicant respectfully requests favorable consideration of these factors in adjudicating the N‑400 naturalization application.

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Applicant’s Name]**


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
            name="RFE Response Brief Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating an Good Moral Character Brief for a Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ```
                # RE: NOTICE OF CONTINUANCE (FORM N-400)

                **Applicant:** [Full Name of Applicant]  
                **Alien Number (A#):** [A-Number]
                
                Dear Sir or Madam,
                
                The United States Citizenship and Immigration Services (USCIS) issued a **Notice of Continuance** for the above-referenced case on **[Date of Original Notice]**. This response is being submitted pursuant to the extended timelines allowed by USCIS based on its flexibility policy announced on **[Flexibility Announcement Date]**.
                
                USCIS allows a response to be submitted within 60 calendar days after the deadline set forth in the original Notice of Continuance, provided the issuance date on the notice falls between **[Start Date]** and **[End Date]**, inclusive. The original notice in this case was issued on **[Date of Issuance]**, and therefore this response is timely.
                
                ## Reason for Continuance:
                
                USCIS records indicate that the applicant traveled outside the United States for a period of six months or more following lawful admission for permanent residence. As such, additional documentation is required to demonstrate that the applicant **did not abandon U.S. residency**.
                
                ## Enclosed Documentation:
                
                1 Copy of [Year] Tax Return  
                2 Notice of Immigrant Visa Case Creation  
                3 Affidavit from [Name or Relationship] affirming that [Affiant] maintained the applicant’s property during the applicant’s absence from the U.S.  
                4 Affidavit from [Name or Relationship] affirming that [Affiant] maintained the applicant’s property during the applicant’s absence from the U.S.
                
                Thank you for your favorable consideration of the enclosed documents. Should additional information be needed to make a decision on this case, please do not hesitate to contact us.

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Applicant’s Name]**


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
        "naturalization",
        "N-400",
        "n-400",
        "Form N-400",
        "g-28",
        "g-1145",
        "passport",
        "visa-pages",
        "travel-itinerary",
        "lawful permanent resident",
        "continuous residence",
        "physical presence",
        "USCIS",
    ],
    "Exhibit List": [
        "exhibit list",
        "naturalization",
        "N-400",
        "n-400",
        "g-28",
        "g-1145",
        "passport",
        "visa-pages",
        "travel-itinerary",
        "tax returns",
        "residence evidence",
        "good moral character",
        "supporting documents",
    ],
    "Eligibility-Checklist Memo": [
        "eligibility checklist",
        "naturalization",
        "N-400",
        "lawful permanent resident",
        "continuous residence",
        "physical presence",
        "travel history",
        "travel-itinerary",
        "passport",
        "visa-pages",
        "residence",
        "USCIS",
    ],
    "Good Moral Character Brief": [
        "good moral character",
        "naturalization",
        "N-400",
        "tax return",
        "criminal history",
        "selective service",
        "family support",
        "travel history",
        "passport",
        "visa-pages",
        "USCIS",
    ],
    "RFE Response Brief": [
        "request for evidence response",
        "RFE",
        "naturalization",
        "N-400",
        "USCIS",
        "continuous residence",
        "physical presence",
        "good moral character",
        "travel-itinerary",
        "passport",
        "visa-pages",
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