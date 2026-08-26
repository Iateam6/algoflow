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
                You are tasked with generating a cover letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):

                ```
                Premium Processing  
                USCIS [Service Center Name]  
                [Street Address]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE:  Request for First Preference Reentry Permit 
                    Self‑Petitioner: [Beneficiary’s Full Name]  
                    Position/Title: [e.g., EVP of Technology]  

                Dear Immigration Officer:

                Please find enclosed the immigrant petition filed on behalf of [Beneficiary’s Full Name] as an Reentry Permit Alien of Extraordinary Ability.

                The following items are included in support of this petition:

                1.  A check for $[Amount] for the I‑907 premium processing fee  
                2.  A check for $[Amount] for the I‑140 filing fee  
                3.  Form I‑907, “Request for Premium Processing Service”  
                4.  Form G‑28, “Notice of Entry of Appearance as Attorney or Accredited Representative”  
                5.  Form I‑140, “Immigrant Petition for Alien Worker”  
                6.  [Beneficiary’s Last Name]’s biographical documents:  
                    a. Passport biographical page  
                    b. O‑1 approval notice  
                    c. Most recent I‑94  
                7.  Attorney’s letter of support  
                8.  Exhibit list  
                9.  Exhibits evidencing [Beneficiary’s Last Name]’s Reentry Permit credentials  

                We respectfully submit that the enclosed documentation establishes [Beneficiary’s Full Name]’s internationally recognized achievements and abilities as a leader in the [field] industry, and that he/she is among the small percentage of individuals who have risen to the top of his/her field.  [Beneficiary’s Last Name] therefore merits classification as an Alien of Extraordinary Ability.

                If you require any further information or documentation to support the attached petition, please do not hesitate to contact our office.

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
         "Support Letter": DocumentPrompt(
            name="Support Letter",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a support letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                [Letterhead or Law Firm Name]  
                [Address Line 1]  
                [Address Line 2]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE:Reentry Permit Petition of [Beneficiary’s Full Name]  

                Dear Immigration Officer:

                Below please find our organized presentation of evidence in support of [Beneficiary’s Full Name]’s classification as an Alien of Extraordinary Ability under 8 C.F.R. § 204.5(h).  Each section corresponds to one of the nine regulatory criteria:

                1. **Documentation of Receipt of Lesser Nationally or Internationally Recognized Prizes or Awards for Excellence**  
                – [Describe awards, dates, issuing organizations, and why they qualify.]

                2. **Documentation of Membership in Associations in the Field Which Require Outstanding Achievements**  
                – [List associations, membership criteria, and evidence of selection.]

                3. **Published Material About the Beneficiary in Professional or Major Trade Publications or Media**  
                – [Cite articles, dates, outlets, and excerpts relevant to the field.]

                4. **Evidence of Participation, Either Individually or on a Panel, as a Judge of the Work of Others**  
                – [Detail panels, dates, selection process, and scope of judging.]

                5. **Evidence of Original Contributions of Major Significance to the Field**  
                – [Summarize innovations, adoption by peers, citation metrics, and impact.]

                6. **Authorship of Scholarly Articles in Professional Journals or Other Major Media**  
                – [List publications, co‑authors, journal impact factors, and download/citation counts.]

                7. **Display of the Beneficiary’s Work at Artistic Exhibitions or Showcases**  
                – [Identify exhibitions or screenings, dates, venues, and audience reach.]

                8. **Evidence That the Beneficiary Has Performed in a Leading or Critical Role for Organizations or Establishments with a Distinguished Reputation**  
                – [Name companies or projects, describe role, and point to recognition.]

                9. **Evidence of Commercial Success in the Performing Arts, as Shown by Box‑Office Receipts or Record, Cassette, Compact Disc, or Video Sales**  
                – [Provide revenue figures, chart positions, and distributor confirmations.]

                **Conclusion & Prayer for Relief**  
                Based on the foregoing evidence (Sections 1–9), [Beneficiary’s Full Name] clearly meets the Reentry Permit criteria for extraordinary ability.  We respectfully request that USCIS grant approval of the Form I‑140 petition.

                If you require further information or documentation, please contact our office.

                **Sincerely,**

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
        "Recommendation Letter": DocumentPrompt(
            name="Recommendation Letter",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Recommendation Letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                U.S. Citizenship and Immigration Services  
                U.S. Department of Homeland Security  

                Date: [YYYY‑MM‑DD]  

                RE:  [Recommender’s Name]’s Recommendation for Reentry Permit Petition of [Beneficiary’s Full Name]  

                Dear Sir or Madam:

                My name is [Recommender’s Name], [Title/Role] at [Organization(s)] and creator/executive producer of [List of Major Works].  I write in strong support of [Beneficiary’s Full Name]’s petition as an individual of extraordinary ability.

                Paragraph 1: Introduce your credentials and relationship to the Beneficiary.

                Paragraph 2: Summarize Beneficiary’s most significant U.S. achievements—lead roles, awards, box‑office metrics, publications, etc.

                Paragraph 3: Highlight Beneficiary’s industry impact (e.g., teaching, guest‑lecturing, innovation in distribution or production).

                Paragraph 4: Conclude that [Beneficiary’s Last Name] clearly qualifies for EB‑1A and that U.S. interests will be served by granting the visa.  Offer to provide additional information if needed.

                Sincerely,

                \_\_\_\_\_\_\_\_\_\_\_,  
                [Recommender’s Printed Name]    
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
                You are tasked with generating a Exhibit List in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                                                Exhibit List  
                                    Self‑Petitioner: [Beneficiary’s Full Name]  
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
        "RFE Response Brief": DocumentPrompt(
            name="RFE Response Brief",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a RFE Response Brief in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Response to USCIS RFE – Reentry Permit for [Applicant’s Full Name]
                **Consulate/USCIS Lockbox Name**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Response to Request for Evidence (RFE) – Reentry Permit for [Applicant’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction & RFE Reference**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”) submits this Response to the Request for Evidence (RFE) issued in connection with the Reentry Permit application (Form I‑131).”  
                **RFE Details:**  
                - Receipt Number: _[Number]_  
                - RFE Issued: _[Date]_  
                - Response Deadline: _[Date]_  
                **Summary of USCIS Concerns:**  
                1. Eligibility under INA § 216(a) and statutory basis.  
                2. Evidence of continuous U.S. residence and non‑abandonment.  
                3. Adequacy of documentation supporting purpose and duration of travel.  

                **Rebuttal to USCIS Concerns**  
                **Concern 1: Statutory Eligibility**  
                - **USCIS Position:** “Applicant may not qualify under INA § 216(a) for a reentry permit.”  
                - **Rebuttal:**  
                The Applicant is a Lawful Permanent Resident (A‑Number [#]), holding a valid PR card through [date], and meets all criteria under INA § 216(a) for issuance of a reentry permit to preserve residence abroad.  

                **Concern 2: Continuous U.S. Residence**  
                - **USCIS Position:** “Insufficient evidence that Applicant maintains continuous U.S. residence.”  
                - **Rebuttal:**  
                Enclosed are: Form I‑94 copy (Ex. B), tax returns for the past [#] years (Ex. G), employment verification letters (Ex. H), and evidence of property and family ties in the U.S. demonstrating non‑abandonment.  

                **Concern 3: Purpose and Duration of Travel**  
                - **USCIS Position:** “Purpose of travel and intended length abroad not adequately supported.”  
                - **Rebuttal:**  
                Applicant’s detailed travel explanation letter (Ex. D) outlines an intended absence of approximately [# months/years] for [reason: e.g., family emergency, employment assignment].  Return‑ticket reservation or proof of ongoing ties (Ex. E) further demonstrates intent to return.  

                **Additional Legal Authority:**  
                Cite INA § 216(a) permitting reentry permits for LPRs; 8 C.F.R. § 223.2 outlining evidentiary requirements.  

                **Supporting Evidence & Exhibits**  
                - **Exhibit A:** Copy of Permanent Resident Card  
                - **Exhibit B:** Copy of Form I‑94  
                - **Exhibit C:** Two passport‑style photos (USCIS specifications)  
                - **Exhibit D:** Letter from Applicant explaining purpose and duration of travel  
                - **Exhibit E:** Return‑ticket reservation or proof of U.S. ties  
                - **Exhibit F:** Prior reentry permits or boarding foils (if any)  
                - **Exhibit G:** Federal tax returns for years [YYYY–YYYY]  
                - **Exhibit H:** Employment verification letter(s)  

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing evidence and statutory authority, the Applicant clearly qualifies for issuance of a Reentry Permit under INA § 216(a).”  
                **Request for Adjudication:** “Applicant respectfully requests prompt adjudication and approval of the Reentry Permit application.”  
                **Point of Contact:** “For any further questions or additional documentation, please contact [Attorney/Representative Name], [Title], at [Phone Number] or [Email Address].”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_, 
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  
 

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
        "Demand Letter": DocumentPrompt(
            name="Demand Letter",
            template=(
                rf"""
                    Today’s date is {current_date}.  
                    You are tasked with generating a Demand Letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                    **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                    - Applicant personal details (name, A‑number, date of birth, country of nationality)
                    - Consulate/USCIS lockbox address and beneficiary’s mailing address
                    - Travel dates, intended length of absence, and reason for travel
                    - Attorney/representative details (if any)

                    **Required Forms**  
                    • Form I‑131, Application for Travel Document (USCIS)  
                    • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                    • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                    • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                    **Supporting Documents**  
                    **Must Have**  
                    • Copy of Permanent Resident Card (USCIS)  
                    • Two passport‑style photos (USCIS photo specs)  
                    • Form I‑94 copy (CBP)  
                    • Passport biographic and visa pages (State/CBP)  
                    • Explanation letter from applicant detailing purpose and length of intended absence  
                    **As Available**  
                    • Return‐ticket reservation or proof of strong ties to the U.S.  
                    • Prior boarding foils or parole docs (if any)  
                    • Evidence of ongoing employment, family, or property in the U.S.  

                    **Step 2**: Use the following structure (raw Markdown, no code fences):

                    ```
                    #Demand for Adjudication Under the Mandamus Act and Administrative Procedure Act – [Applicant’s Full Name]  
                    **[Attorney’s Name]**  
                    **[Law Firm Name]**  
                    **[Street Address]**  
                    **[City, State, ZIP Code]**  
                    **[Phone Number]**  
                    **[Email Address]**  

                    **Date:** _YYYY-MM-DD_  

                    **RE:** _Demand for Adjudication under the Mandamus Act – Reentry Permit (Form I‑131) for [Applicant’s Full Name]_  

                    ### Dear USCIS Officer,  

                    **Introduction & Jurisdiction**  
                    - **Parties:** “This letter is submitted by **[Law Firm Name]** on behalf of **[Applicant’s Full Name]** (the “Applicant”), in support of the Reentry Permit (Form I‑131) application filed on [I‑131 Filing Date].”  
                    - **Procedural History:**  
                    - I‑131 Filed: _[Date]_; Receipt No.: _[Number]_  
                    - RFE Issued (if any): _[Date]_ → Response Filed: _[Date]_  
                    - Current Delay: _[Number]_ days beyond USCIS’s published 90‑day processing guideline  
                    - **Jurisdiction:** Demand is made under *28 U.S.C. § 1361* (mandamus) and *5 U.S.C. § 555(b)* (unreasonable delay).  

                    **Factual Background**  
                    - **Applicant Status:** Lawful Permanent Resident, A‑Number [#], PR card valid through [Date].  
                    - **Purpose of Travel:** [Describe reason—e.g., family emergency, employment abroad, education].  
                    - **Intended Absence:** Approximately [# months/years], departing [Departure Date], returning by [Expected Return Date].  
                    - **Procedural Compliance:** All required forms and fees submitted in accordance with USCIS guidelines.  

                    **Legal Standard for Mandamus**  
                    - **Clear Right:** Applicant’s undisputed right to timely adjudication of Form I‑131.  
                    - **Non‑Discretionary Duty:** USCIS must process travel‑document applications within a reasonable period.  
                    - **No Adequate Alternative:** Inquiries and service requests have not yielded a decision.  
                    - **Agency Guidelines:** USCIS processing goal is 90 days for Form I‑131 (see USCIS Processing Times webpage).  

                    **Demand for Relief**  
                    - **Relief Sought:** Final adjudication of the Reentry Permit application within **14 days** of receipt.  
                    - **Statutory Authority:** *28 U.S.C. § 1361*; *5 U.S.C. § 555(b)*.  
                    - **Consequences if Unresolved:**  
                    - **Applicant Hardship:** Risk of abandonment of residency, inability to reenter U.S. after [Departure Date].  
                    - **Family & Employment Impact:** Disruption of employment, education of dependents, and personal obligations.  

                    **Prejudice & Hardship**  
                    - **Applicant Impact:**  
                    - Potential loss of LPR status if permit not issued prior to departure.  
                    - Emotional and financial strain on family left in the U.S.  
                    - **Irreparable Injury:** Monetary damages inadequate; only mandamus relief will preserve Applicant’s right to reentry.  

                    **Exhibits & Supporting Documents**  
                    - **Ex. A:** Complete Form I‑131 petition package  
                    - **Ex. B:** USCIS receipt notice for Form I‑131  
                    - **Ex. C:** Copy of Permanent Resident Card  
                    - **Ex. D:** Two passport‑style photos  
                    - **Ex. E:** Applicant’s letter explaining purpose and duration of travel  
                    - **Ex. F:** Copy of Form I‑94  
                    - **Ex. G:** Passport biographic and visa pages  

                    **Conclusion & Next Steps**  
                    - **Final Demand:** “We request USCIS issue a final decision on Form I‑131 no later than 14 days from service of this demand.”  
                    - **Service Confirmation:** “Please confirm receipt via email to **[Attorney’s Email]** or fax to **[Fax Number]**.”  
                    - **Litigation Warning:** “Absent timely action, we will file a Writ of Mandamus in the U.S. District Court for the District of **[District]**, and seek appropriate fees and costs under the Equal Access to Justice Act.”  
                    - **Attorney Availability:** “[Attorney’s Name] is available to provide any further information or documentation USCIS may require.”  

                    **Sincerely,**  
                    \_\_\_\_\_\_\_\_\_\_\_  
                    **[Attorney’s Full Name], Esq.**  
                    **[Law Firm Name]**  



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
        "Assessment Report": DocumentPrompt(
            name="Assessment Report",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Assessment Report in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Reentry Permit Support Letter – [Applicant’s Full Name]
                **[Consulate/USCIS Lockbox Name]**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Support Letter for Reentry Permit (Form I‑131) – [Applicant’s Full Name]  

                **Dear Consular Officer/USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”), hereby submits this support letter in conjunction with Form I‑131, Application for Travel Document, to secure a Reentry Permit under INA § 216(a).”  
                **Purpose of Travel:** To permit temporary departure from the United States for [reason for travel] and return within [intended length of absence].  

                **Background & Eligibility**  
                **Residency Status:** Applicant is a Lawful Permanent Resident (A‑Number [#]), holding a Permanent Resident Card valid through [date].  
                **Statutory Basis:** Requested under INA § 216(a) to maintain residence status while traveling abroad for [duration].  
                **Travel Details:** Departure on [departure date]; expected return by [return date].  

                **Required Forms & Evidence**  
                - **Form I‑131:** Completed and signed.  
                - **Form G‑1145:** E‑Notification of Application/Petition Acceptance.  
                - **Form G‑28:** Notice of Entry of Appearance as Attorney (if represented).  
                - **DS‑160:** Confirmation page (if consular processing required).  

                **Supporting Documents Provided**  
                1. Copy of Permanent Resident Card  
                2. Two passport‑style photos (USCIS specifications)  
                3. Copy of Form I‑94  
                4. Passport biographic and visa pages  
                5. Letter from Applicant explaining purpose and duration of travel  
                6. Proof of return‑ticket reservation or other evidence of U.S. ties 

                **Procedural Compliance**  
                All documentation complies with USCIS and Department of State requirements. The Applicant’s continuous residence in the United States is established, and no abandonment of status will occur.  

                **Conclusion & Request**  
                “Based on the foregoing, the Applicant respectfully requests prompt adjudication and issuance of the Reentry Permit. Please contact **[Attorney/Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]** for any questions or additional documentation.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_\_  
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  

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
                You are tasked with generating a Eligibility Memorandum in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Reentry Permit Support Letter – [Applicant’s Full Name]
                **Consulate/USCIS Lockbox Name**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Eligibility Assessment Report for Reentry Permit (Form I‑131) – [Applicant’s Full Name]  

                **Dear Consular Officer/USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”) respectfully submits this Eligibility Assessment Report in support of the Reentry Permit (Form I‑131) application under INA § 216(a) to permit travel abroad for [intended length of absence] beginning [departure date].”  
                **Travel Overview:** Departure: _[Departure Date]_; Return: _[Expected Return Date]_; Purpose: _[Reason for Travel – e.g., family emergency, employment, education]_  

                **Statutory Basis**  
                Requested under INA § 216(a), which authorizes Lawful Permanent Residents to apply for a Reentry Permit to preserve continuous residency while abroad.  

                **Eligibility Evaluation**  
                - **Residency Status:** Applicant is a Lawful Permanent Resident, A‑Number [#], with Form I‑551 valid through [PR Card Expiration Date].  
                - **Non‑Abandonment of Residence:** Applicant maintains U.S. ties through ongoing employment ([Employer Name]), family ([Spouse/Children]), and property ownership ([Address or Description]).  
                - **Purpose & Duration:** Travel for [detailed purpose], with planned absence of approximately [# months/years], returning by [Expected Return Date].  
                - **Compliance with Requirements:** Applicant has not received prior removal proceedings or reentry refusals and has no known inadmissibility issues.  

                **Supporting Evidence & Exhibits**  
                **Exhibit A:** Copy of Permanent Resident Card (Form I‑551)  
                **Exhibit B:** Two passport‑style photos (USCIS specifications)  
                **Exhibit C:** Copy of Form I‑94 Arrival/Departure Record  
                **Exhibit D:** Passport biographic and visa pages  
                **Exhibit E:** Letter from Applicant explaining purpose and duration of travel  
                **Exhibit F:** Return‑ticket reservation or proof of ties to the U.S.
                **Exhibit G:** Prior boarding foils or parole documents.  
                **Exhibit H:** Evidence of ongoing U.S. ties (employment, family, property)

                **Procedural Compliance**  
                All forms and evidence have been compiled in strict accordance with USCIS and Department of State guidelines. The Applicant’s continuous U.S. residency is well‑documented, and all statutory and regulatory criteria have been satisfied.  

                **Conclusion & Request**  
                “Based on the foregoing analysis and supporting exhibits, the Applicant clearly meets the eligibility requirements for issuance of a Reentry Permit under INA § 216(a). We respectfully request prompt adjudication of this application. Please contact [Attorney/Representative Name] at [Phone Number] or [Email Address] for any further inquiries or documentation.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_\_ 
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  

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
        "Visa Application Summary Report": DocumentPrompt(
            name="Visa Application Summary Report",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Visa application in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Reentry Permit Application Summary Report – [Applicant’s Full Name]
                **Consulate/USCIS Lockbox Name**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Reentry Permit Application Summary Report – [Applicant’s Full Name]  

                **Dear Consular Officer/USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”) submits this Summary Report in support of the Reentry Permit application (Form I‑131) under INA § 216(a).”  
                **Travel Overview:** Intended departure: _[Departure Date]_ from _[U.S. Port of Exit]_ and return by _[Expected Return Date]_.  
                **Objective:** Provide a concise narrative of the key evidence demonstrating the Applicant’s eligibility and continuity of Lawful Permanent Resident status.

                **Summary of Evidence**  
                The record clearly establishes that the Applicant maintains continuous U.S. residency and meets all statutory requirements for a Reentry Permit. A copy of the Permanent Resident Card valid through _[Date]_ confirms status continuity. Two passport‑style photos in compliance with USCIS specifications and the completed Form I‑131 demonstrate procedural correctness. The I‑94 arrival/departure record and biographic pages of the passport corroborate travel history. The letter from the Applicant detailing purpose (_[Reason for Travel]_) and intended absence of _[Length of Time]_ evidences bona fide intent to return. Supporting exhibits such as return‑ticket reservation and proof of U.S. ties (employment letter, property deeds) further substantiate non‑abandonment of residence.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing, the Applicant satisfies all requirements under INA § 216(a) for issuance of a Reentry Permit.”  
                **Request for Adjudication:** “Applicant respectfully requests prompt adjudication and approval of the Reentry Permit application.”  
                **Point of Contact:** “For any questions or additional documentation, please contact **[Attorney/Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_  
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  

                ```

                Step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
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
        "reentry permit",
        "form-i-131",
        "Form I-131",
        "Application for Travel Document",
        "form-g-1145",
        "form-g-28",
        "permanent-resident-card",
        "green card",
        "Form I-551",
        "passport biographic page",
        "return-ticket-reservation",
        "proof of ties",
        "USCIS filing",
    ],
    "Support Letter": [
        "reentry permit support letter",
        "lawful permanent resident",
        "continuous residence",
        "non-abandonment",
        "explanation-for-extended-travel",
        "purpose of travel",
        "intended absence",
        "employment",
        "family",
        "property",
        "attorney",
        "representative",
    ],
    "Recommendation Letter": [
        "recommendation letter",
        "reentry permit",
        "lawful permanent resident",
        "purpose of travel",
        "intended absence",
        "continuous residence",
        "non-abandonment",
        "employment ties",
        "family ties",
        "property ties",
        "supporting evidence",
    ],
    "Exhibit List": [
        "exhibit list",
        "form-i-131",
        "Form I-131",
        "form-g-1145",
        "form-g-28",
        "permanent-resident-card",
        "Form I-551",
        "passport",
        "biographic page",
        "visa pages",
        "explanation-for-extended-travel",
        "return-ticket-reservation",
        "proof of ties",
    ],
    "RFE Response Brief": [
        "request for evidence response",
        "RFE",
        "USCIS",
        "reentry permit",
        "Form I-131",
        "lawful permanent resident",
        "continuous residence",
        "non-abandonment",
        "purpose of travel",
        "intended absence",
        "permanent resident card",
        "proof of ties",
    ],
    "Demand Letter": [
        "demand letter",
        "pending adjudication",
        "adjudication delay",
        "USCIS",
        "reentry permit",
        "Form I-131",
        "filing receipt",
        "form-g-28",
        "attorney",
        "representative",
        "final decision",
    ],
    "Assessment Report": [
        "assessment report",
        "reentry permit",
        "Form I-131",
        "lawful permanent resident",
        "continuous residence",
        "non-abandonment",
        "purpose of travel",
        "intended absence",
        "permanent resident card",
        "forms and exhibits",
        "proof of ties",
        "USCIS",
    ],
    "Eligibility Memorandum": [
        "eligibility memorandum",
        "reentry permit",
        "Form I-131",
        "Application for Travel Document",
        "lawful permanent resident",
        "green card",
        "Form I-551",
        "continuous residence",
        "non-abandonment",
        "explanation-for-extended-travel",
        "purpose of travel",
        "intended absence",
        "employment",
        "family",
        "property",
    ],
    "Visa Application Summary Report": [
        "visa application summary report",
        "reentry permit",
        "form-i-131",
        "Form I-131",
        "Application for Travel Document",
        "form-g-1145",
        "form-g-28",
        "permanent-resident-card",
        "passport",
        "biographic page",
        "visa pages",
        "return-ticket-reservation",
        "proof of ties",
        "purpose of travel",
        "intended absence",
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