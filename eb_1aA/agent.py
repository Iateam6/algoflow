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
    """Build the prompt registry for each EB-1A I-140 with Sponsors output document."""
    # Get today’s date in the desired format
    current_date = format_current_date()

    return {
        "Form I-140 Cover Letter": DocumentPrompt(
            name="Form I-140 Cover Letter",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a I-140 Cover Letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ``` 
                **[Date]**
                
                **U.S. Port of Entry / USCIS Address**
                **NAFTA Free Trade Examiner**

                ### **RE: EB-1A I-140 with Sponsors Application**

                **Employer:** [Company Name]<br>
                **Beneficiary:** [Beneficiary Full Name]<br>
                **Position:** [Job Title]<br>
                **Country:** [Country name of the beneficiary]


                Dear Free Trade Examiner:

                This letter is being submitted in support of **[Beneficiary Full Name]**’s EB-1A I-140 with Sponsors application. **[He/She/They]** is a **[Canadian/Mexican]** national seeking EB-1A status as a **[EB-1A Profession]** under the provisions of the USMCA.

                **[Beneficiary Full Name]** qualifies for EB-1A classification by virtue of **[his/her/their]** **[Degree]** in **[Field of Study]** and substantial experience in the field. **[He/She/They]** will be employed by **[Company Name]** in the position of **[Job Title]**, where **[he/she/they]** will **[brief description of duties — 2-3 sentences]**.

                **[Beneficiary Full Name]** will receive an annual compensation of **$[Annual Salary]**. The employment is expected to begin on **[Start Date]** and continue for up to three years. **[He/She/They]** has declared the intent to depart the United States upon completion of the authorized period of stay.

                **Enclosed, please find the following materials in support of this application:**

                [Add the provided exhibit here with a short description like bellow
                EXAMPLE;
                Exhibit 1:	Form G-1450 in the amount of $2,965 for the I-907 Premium Processing fee
                            Form G-1450 in the amount of $510 for the I-129 Non-Immigrant Petition
                            Form G-1450 in the amount of $300 for the Asylum fee.
                            Form G-28 “Notice of Entry of Appearance as Attorney”
                            Form I-907 “Request for Premium Processing”
                            Form I-129 + TN Supplement “Non-Immigrant Petition for Alien Workers”
                Exhibit 2:	Copy of Applicant’s Canadian Passport as proof of his Canadian citizenship, proof of lawful stay in the U.S., and proof of ties in home country.
                Exhibit 3:   Letter of support and explanation for EB-1A qualification
                Exhibit 4:   Copy of Applicant’s educational evaluation, diploma, and transcripts
                Exhibit 5:   Copy of Applicant’s resume showing his work experience
                Exhibit 6:   Job Offer Letter with Job Description
                Exhibit 7:   Information about the Employer
                NOTE: Do not use this in the generate file covers USE the provided one
                ]

                Thank you for your assistance. Should you require any additional information, please do not hesitate to contact our office.

                **Very truly yours,**

                **[Preparer/Lawyer's Full Name], Esq.**  
                **[Firm Name]** 
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Exhibits", list only the exhibits from supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits and the numbering of exhibits should start from 1 going on. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
        ),
        "Intent to Depart": DocumentPrompt(
            name="Intent to Depart Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a intent to depart letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                **Petitioner:** [Company Name]  
                **Applicant:** [Mr./Ms.] [Beneficiary Full Name]  
                **Nationality:** [Nationality]
                **Classification:** EB-1A I-140 with Sponsors Application

                **USCIS Filing Address / Port of Entry**

                **EXPRESSION OF INTENT TO RETURN TO MY HOME COUNTRY, [COUNTRY NAME], UPON EXPIRATION OF MY EB-1A NON-IMMIGRANT STATUS**

                An applicant who is the beneficiary of a non-immigrant visa petition will need to satisfy the immigration officer that his or her intent is to depart the United States at the end of his or her authorized stay in EB-1A status and not stay in the United States to adjust status or otherwise remain in the United States.

                Below is [Mr./Ms.] **[Beneficiary Full Name]**’s declaration of intent to depart the United States upon completion of his/her authorized stay in EB-1A status.

                **I, [Beneficiary Full Name], do hereby certify, swear, or affirm under penalty of perjury as to the truth of the following statements:**

                1.I am a National and Citizen of **[Country]**. I permanently reside in **[Country of Residence]** at the following address: **[Beneficiary’s Foreign Address]**.

                2.I am applying for EB-1A classification as a **[Job Title]** in **[Company Name]**, a U.S. Treaty Enterprise.

                3.If the United States Citizenship and Immigration Services approves my application, I understand that my stay in the United States is temporary. Thus, I intend to remain in the U.S. strictly through the duration of my authorized stay pursuant to the TN status or any extension thereof. At the end of my authorized stay in TN status, I intend to leave the United States and return to **[Country]**. I do not intend to stay in the United States to adjust status or otherwise remain in the United States regardless of legality of status.

                **Signature:**  
                \_\_\_\_\_\_\_\_\_\_\_,

                **By:** [Beneficiary Full Name]<br>
                **Title:** [Job Title]<br>
                **Company:** [Company Name]<br>
                **Date:** [Date]
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 6.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 7.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 8.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
        ),
        "Support Letter": DocumentPrompt(
            name="Support Letter Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a support letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                **[Date]**

                **United States Customs and Border Protection / USCIS**
                
                ### **RE: Petition for EB-1A I-140 with Sponsors Status for Mr. / Ms. [Beneficiary Full Name]**
                
                **Profession (USMCA, formerly NAFTA):** [Job Title]
                
                Dear Sir or Madam:
                
                This letter is written in support of Mr. / Ms. **[Beneficiary Full Name]**'s petition for EB-1A I-140 with Sponsors status pursuant to the USMCA, formerly North American Free Trade Agreement ("NAFTA"). **[Company Name]** has made an offer of temporary employment to Mr. / Ms. **[Beneficiary Full Name]** in the position of **[Job Title]** (under the USMCA classification of **[Job Title]**) with a monthly base salary of $[Monthly Salary] which translates to **$[Annual Salary]** annually. We are seeking the approval of his/her petition to enable **[Company Name]** to employ him/her immediately upon the approval of his/her EB-1A I-140 with Sponsors visa, up to three years.
                
                ### BACKGROUND OF THE EMPLOYER
                
                **[Company Name]**
                
                **[Company Name]** is **[full company description]**. (See the exhibit(s) containing the employer background information)
                
                ### Job Title
                
                **[Company Name]** has offered Mr. / Ms. **[Beneficiary Full Name]** the position of **[Job Title]** in which capacity he/she will serve at the pleasure and assignment of **[Name of Immediate Supervisor]**. In his/her position he/she will be responsible for the following duties:
                
                **[Duty 1 in bulet point]**
                **[Duty 2 in bulet point]**
                **[Duty 3 in bulet point]**
                **[Duty 4 in bulet point]**
                **[Duty 5 in bulet point]**
                **[Duty 6 in bulet point]**
                **[Duty 7 in bulet point]**
                **[Duty 8 in bulet point]**
                **[Duty 9 in bulet point]**
                **[Duty 10 in bulet point]**
                
                This is a strategic and consultative role designed to support the growth and cohesion of **[Company Name]**'s efforts. (See the exhibit(s) containing the job offer letter, job duties and responsibilities, and consulting agreement)
                
                ### APPLICANT'S QUALIFICATIONS
                
                Mr. / Ms. **[Beneficiary Full Name]** holds a bachelor's degree in **[Degree]** with a major in **[Major]** from **[University Name]**, providing a strong academic foundation. **[His/Her]** career spans diverse leadership roles. **[Brief professional summary]**.
                
                Mr. / Ms. **[Beneficiary Full Name]**'s extensive background in **[key areas]** positions him/her as a highly capable candidate for a **[Job Title]** role. (See the exhibit(s) containing Mr. / Ms. **[Beneficiary Full Name]**'s educational and professional credentials)
                
                ### CONCLUSION
                
                It is evident that Mr. / Ms. **[Beneficiary Full Name]** is eligible for the EB-1A I-140 with Sponsors visa as a **[Job Title]** as he/she possesses the appropriate degree and a considerable number of years of experience. **[Company Name]** offers Mr. / Ms. **[Beneficiary Full Name]** employment immediately upon the approval of his/her EB-1A I-140 with Sponsors visa, up to three years. He/She will be compensated at a monthly base salary of $[Monthly Salary] which translates to **$[Annual Salary]** annually.
                
                For the foregoing reasons, we respectfully request your favorable adjudication of his/her EB-1A I-140 with Sponsors petition on behalf of Mr. / Ms. **[Beneficiary Full Name]**.
                
                **Very truly yours,**
                
                **[Name of Immediate Supervisor]**  
                **[Position]**  
                **[Company Name]**  
                **[Contact Information]**
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 6.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 7.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 8.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
        ),
        "Recommendation-Letter": DocumentPrompt(
            name="Recommendation Letter Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Recommendation Letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                U.S. Citizenship and Immigration Services  
                U.S. Department of Homeland Security  

                Date: [YYYY‑MM‑DD]  

                RE:  [Recommender’s Name]’s Recommendation for EB-1A I-140 with Sponsors Petition of [Beneficiary’s Full Name]  

                Dear Sir or Madam:

                My name is [Recommender’s Name], [Title/Role] at [Organization(s)] and creator/executive producer of [List of Major Works].  I write in strong support of [Beneficiary’s Full Name]’s petition as an individual of extraordinary ability.

                Paragraph 1: Introduce your credentials and relationship to the Beneficiary.

                Paragraph 2: Summarize Beneficiary’s most significant U.S. achievements—lead roles, awards, box‑office metrics, publications, etc.

                Paragraph 3: Highlight Beneficiary’s industry impact (e.g., teaching, guest‑lecturing, innovation in distribution or production).

                Paragraph 4: Conclude that [Beneficiary’s Last Name] clearly qualifies for EB-1A I-140 with Sponsors classification and that U.S. interests will be served by granting the visa.  Offer to provide additional information if needed.

                **very truly yours,**

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
            name="Exhibit List Agent",
            template=(
                rf"""

                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                **List of Supporting Documents**  
                **EB-1A I-140 with Sponsors Visa Application**

                **Petitioner:** [COMPANY NAME]  
                **Beneficiary:** [BENEFICIARY’S NAME]

                Pursuant to the United States-Mexico-Canada Agreement, a foreign national is entitled to enter the United States under the TN Status Visa category. Below is a complete list of supporting documents submitted to establish that **Mr. / Ms. [BENEFICIARY’S NAME]**, a citizen of **[Canada / Mexico]**, is qualified for the EB-1A Visa.

                | **Exhibit 1** | **Forms & Fees** |
                |---------------|------------------|
                |               | 1.1 Cover Letter |
                |               | 1.2 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.3 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.4 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.5 Form G-28 “Notice of Entry of Appearance as Attorney” |
                |               | 1.6 Form I-907 “Request for Premium Processing” |
                |               | 1.7 Form I-129 + EB-1A I-140 with Sponsors Supplement “Petition for a Nonimmigrant Worker” |
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
        "Evidence-Summary Chart": DocumentPrompt(
            name="Evidence-Summary Chart Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Evidence Summary Chart for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                ## Evidence‑Summary Chart

                **1. Awards and Recognitions**  
                Document any nationally or internationally recognized awards that the beneficiary has received. Include details such as the awarding body, date, and context. Evidence may also include team awards, provided the beneficiary is a named recipient. Such awards demonstrate peer or institutional validation of excellence. If the award is part of a competition or grant selection process, explain its selectivity and prestige. This evidence helps satisfy the criterion under 8 CFR § 204.5(h)(3)(i) by validating recognized achievement in the field. Ensure documentation shows the beneficiary’s direct award receipt.

                **2. Membership in Associations**  
                Provide proof of membership in professional associations that require outstanding achievement for admission. Include documentation of the association’s selective criteria, the beneficiary’s membership status, and any leadership or committee roles held. Membership alone is assessable but becomes stronger when supplemented by active engagement such as organizing events or chapter leadership. This qualifies under the membership criterion and may bolster other criteria like critical roles or judging.

                **3. Published Material About the Beneficiary**  
                Include copies or clippings of media articles—preferably in major media outlets or professional trade journals—that discuss the beneficiary’s work in depth. The material must focus on the beneficiary and their contributions, not simply a passing mention. Provide full citations including title, author, date, publication, and translations if necessary. This aligns with the published material criterion and demonstrates external validation of the beneficiary’s influence.

                **4. Judging or Reviewing of Others’ Work**  
                Demonstrate instances where the beneficiary served as a peer reviewer, judging panel member, grant evaluator, or academic organizer. Include invitations, confirmation letters, or membership rosters listing the beneficiary’s participation. Even informal or one-time invites can qualify, provided they show recognition of expertise. This evidence supports the criterion under judging of others’ work. It further signals the beneficiary’s stature in their field.

                **5. Original Contributions of Major Significance**  
                Document contributions that have measurably impacted the field—such as new methodologies, patents, breakthrough research, or technological innovations. Provide supporting letters from experts describing the significance, peer citations, adoption rate, or real‑world impact. This category reflects the beneficiary’s role in advancing the discipline and satisfies the contributions criterion under the regulatory standard.

                **6. Authorship of Scholarly Articles**  
                List authored scholarly articles in reputable, peer‑reviewed journals or widely recognized professional publications. Include author order, impact factor of the journal, citation count, and audience reach. If the beneficiary was lead or sole author on high‑impact work, emphasize that. Even a single well‑cited article may qualify; but context on its prestige strengthens the case.

                **7. Leading or Critical Role in Distinguished Organizations**  
                Show evidence that the beneficiary held a leading or essential role in institutions or projects with reputational prominence. Documents may include organizational charts, reference letters, performance evaluations, or press coverage highlighting the beneficiary’s impact. Include details on how their contributions were central to organizational success. This supports the critical role criterion, signaling professional leadership.

                **8. High Salary or Commercial Success** *(if applicable)*  
                If relevant, provide documentation demonstrating that the beneficiary commands a high salary or achieved notable commercial success compared to peers. This may include salary records, contracts, revenue figures, or royalty statements in performing arts or innovations. Demonstrating remuneration above field norms supports eligibility under high‑salary or commercial success criteria.

                **9. Comparable Evidence** *(if standard criteria don’t directly fit)*  
                Where conventional categories may not apply—such as entrepreneurial achievements or field‑specific milestones—provide alternative evidence of comparable significance. Examples include unique funding awards, high‑profile conference presentations, invited keynote speeches, or patents obtained by startups. The updated Policy Manual explicitly allows such flexibility, particularly in STEM or innovative fields. Explain the relevance clearly.

                **Final Merits Commentary**  
                Summarize how the totality of submitted evidence demonstrates that the beneficiary is among the small percentage at the very top of their field with sustained national or international acclaim. Explain how meeting three or more regulatory criteria, along with the overall depth and quality of documentation, satisfies the two‑step adjudicative test under USCIS standards.
                

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
                You are tasked with generating a RFE Response Brief for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                # Response to USCIS RFE – EB-1A I‑140 Petition for [Beneficiary’s Full Name]  
                **[Self‑Petitioner Name or Representative]**  
                **[Address]**  
                **[City, State, ZIP Code]**

                **Date:** _YYYY‑MM‑DD_

                **Subject:** Response to Request for Evidence (RFE) – EB‑1A Petition for [Beneficiary’s Full Name]

                **Dear USCIS Officer,**

                **Introduction & RFE Reference**  
                **Self‑Petitioner:** “[Beneficiary’s Full Name], as beneficiary and self‑petitioner, submits this response to the RFE issued regarding the Form I‑140 petition under the EB‑1A category.”  
                **RFE Details:**  
                   Receipt Number: _[Number]_  
                   RFE Issued: _[Date]_  
                   Response Deadline: _[Date]_  
                **Summary of USCIS Concerns:**  
                   1. Meeting at least three criteria under 8 C.F.R. § 204.5(h).  
                   2. Evidence of sustained national or international acclaim.  
                   3. Proof of intent to continue work in the field of extraordinary ability.

                **Response to USCIS Concerns**  
                **Concern 1: Satisfaction of Regulatory Criteria**  
                **USCIS Position:** “Insufficient documentation demonstrating satisfaction of a minimum of three alternative criteria for EB‑1A eligibility.”  
                **Rebuttal:**  
                We submit enhanced evidence addressing each criterion claimed: notably  
                • Receipt of lesser nationally‑recognized prizes (Ex. A);  
                • Authorship of scholarly articles in reputable journals (Ex. B);  
                • Original contributions of major significance supported by expert letters (Ex. C).  
                These documents are mapped precisely to each regulatory criterion under 8 C.F.R. § 204.5(h) and include explanatory summaries.

                **Concern 2: Sustained National or International Acclaim**  
                **USCIS Position:** “Documents do not clearly establish sustained acclaim or recognition at the top of the field.”  
                **Rebuttal:**  
                We include independent expert letters from recognized authorities in the field, highlighting the beneficiary’s international prominence, citation record, presentation invitations, and media coverage (Ex. D). Publications and citation metrics are arrayed chronologically to show consistent recognition over time in line with AAO precedent.  

                **Concern 3: Intent to Continue Work in Expertise**  
                **USCIS Position:** “Lack of evidence confirming that the beneficiary intends to continue working in the field of extraordinary ability in the United States.”  
                **Rebuttal:**  
                We have enclosed a detailed statement by the beneficiary outlining future research or professional plans, supported by letters from collaborators or institutions (Ex. E), documenting concrete plans to continue impactful work in the same field in the U.S. This satisfies the continuing‐work requirement and aligns with USCIS RFE guidance.

                **Additional Legal References**  
                We cite INA § 203(b)(1)(A) and 8 C.F.R. § 204.5(h) as governing authority for extraordinary ability classification.

                **Conclusion & Request**  
                **Factual and Regulatory Basis:** “Based on the robust documentary evidence and expert declarations submitted, the beneficiary satisfies at least three regulatory criteria under 8 C.F.R. § 204.5(h), demonstrates sustained acclaim, and intends to continue in the field.”  
                **Request for Approval:** “The petitioner respectfully requests prompt approval of the EB‑1A I‑140 petition for [Beneficiary’s Full Name].”  
                **Point of Contact:** “Please contact **[Representative Name]**, **[Title/Role]**, at **[Phone Number]** or **[Email Address]** for any further information or documentation.

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_\_\_,  
                **[Representative Name], [Title]**  
                **[If self‑petition: Beneficiary Name]**

  

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
            name="Demand Letter Agent",
            template=(
                        rf"""
                        Today’s date is {current_date}.  
                        You are tasked with generating a Demand Letter for an EB-1A I-140 with Sponsors visa application.

                        **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                        - Personal details of the beneficiary or the client.
                        - Employer details.
                        - Job description and duties.
                        - Required forms:
                            - Form G-1145, E-Notification of Application/Petition Acceptance
                            - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                            - Form I-907, Request for Premium Processing Service
                            - Form I-140, Immigrant Petition for Alien Worker
                        - Supporting documents:
                            - All degree certificates
                            - Awards and recognitions
                            - Degree evidence
                            - Birth certificate
                            - Form I-94 (Arrival/Departure Record)
                            - Form W-2/1099 (Wage and Tax Statements)
                            - Publications
                            - Membership in organizations
                            - Passport
                            - Visa pages
                            - Letters of recommendation
                            - Formation documents (e.g., Articles of Incorporation)
                            - Federal tax returns
                            - Media reports

                        **Step 2**: Use the following structure for the letter:
    
                        ```
                        # Demand for Adjudication Under the Mandamus Act and Administrative Procedure Act – [Beneficiary’s Full Name]  
                        **[Attorney’s Name]**  
                        **[Law Firm Name]**  
                        **[Street Address]**  
                        **[City, State, ZIP Code]**  
                        **[Phone Number]**  
                        **[Email Address]**  

                        **Date:** _YYYY‑MM‑DD_  

                        **Employer / Sponsor Contact:**  
                        - **Name:** _[Sponsor or Employer’s Name]_  
                        - **Title:** _[Title]_  
                        - **Company:** _[Sponsor/Company Name]_  
                        - **Address:** _[Street Address], City, State, ZIP Code_  

                        **RE:** _Demand for Adjudication under the Mandamus Act – [Beneficiary’s Full Name] (EB‑1A I‑140 with Sponsor)_  

                        ### Dear [Sponsor or Employer Name],  

                        **Introduction & Jurisdiction**  
                        - **Parties:** “This letter is submitted by **[Law Firm Name]** on behalf of **[Sponsor/Employer Name]** (the “Petitioner”) in support of its EB‑1A I‑140 petition for **[Beneficiary Name]** (the “Beneficiary”), including sponsor support documentation.”  
                        - **Procedural History:**  
                        - I‑140 Filed: _[Date]_; Receipt No.: _[Number]_  
                        - RFE Issued (if any): _[Date]_ → Response Filed: _[Date]_ (if applicable)  
                        - Current Delay: _[Number]_ days beyond USCIS’s published processing benchmarks or internal guidance  
                        - **Jurisdiction:** Demand is made under *28 U.S.C. § 1361* (mandamus) and *5 U.S.C. § 555(b)* (unreasonable delay) to compel adjudication of the EB‑1A I‑140 petition.  

                        **Factual Background**  
                        - **Sponsor Profile:** Description of the sponsor/employer’s industry, size, and role in supporting the beneficiary’s extraordinary achievements and ability.  
                        - **Beneficiary Credentials:** Highest degree, field, exceptional achievements, international recognition, published works, citations, awards, former visa/status history, and evidence of sustained acclaim.  
                        - **Position and Sponsorship Details:** Offered position title, SOC code (if applicable), wage level, job duties aligned with extraordinary ability, sponsor’s role in supporting continued acclaim or national/international impact.  
                        - **Key Dates (Timeline):**  
                        - • _[Date]_ – I‑140 Filed  
                        - • _[Date]_ – RFE Issued (Ex. B), if applicable  
                        - • _[Date]_ – RFE Response Filed (Ex. C), if applicable  
                        - • _Today’s Date_ – Over _[X]_ days past reasonable adjudication period  

                        **Legal Standard for Mandamus and APA**  
                        - **Clear Right:** Petitioner and beneficiary have an indisputable right to timely adjudication of the I‑140 under the INA and USCIS regulations.  
                        - **Non‑Discretionary Duty:** USCIS has a ministerial duty to adjudicate filed I‑140 petitions within a reasonable timeframe.  
                        - **No Adequate Alternative Remedy:** Administrative status inquiries and service requests are inadequate under existing case law.  
                        - **Governing Authority:** Clear statutory basis under *28 U.S.C. § 1361* and *5 U.S.C. § 555(b)*; courts regularly permit mandamus or APA relief for delayed petition adjudications.

                        **Demand for Relief**  
                        - **Relief Sought:** We hereby demand that USCIS render a final decision on the EB‑1A I‑140 petition for **[Beneficiary’s Full Name]**, sponsored by **[Sponsor/Employer]**, no later than **14 calendar days** from receipt of this letter.  
                        - **Consequences if Unresolved:**  
                        - **Sponsor Hardship:** Delay imperils the sponsor’s planned engagement with the beneficiary, interrupts project timelines, and risks reputational harm or contractual penalties _(~ $X/week of delay)_.  
                        - **Beneficiary Hardship:** Continued delay causes expiration of existing status, risk of unlawful presence, disruption to international recognition trajectory, and harm to dependents.  

                        **Prejudice & Hardship**  
                        - **Sponsor Impact:**  
                        - Financial losses: ~$[Amount] per week due to postponed contributions or engagements.  
                        - Operational setbacks: inability to leverage beneficiary’s expertise for ongoing or upcoming projects.  
                        - **Beneficiary Impact:**  
                        - Loss of ability to maintain work authorization; risk of falling out of status on _[Date]_.  
                        - Negative consequences to international collaborative work and family stability.  
                        - **Irreparable Injury:** Monetary damages cannot adequately compensate for loss of status or professional momentum; only judicial relief can ensure adjudication.  

                        **Conclusion & Next Steps**  
                        - **Final Demand:** “We request that USCIS issue a final adjudication decision no later than 14 days from service of this demand.”  
                        - **Service Confirmation:** “Please confirm receipt via email to **[Attorney’s Email]** or fax to **[Fax Number]**.”  
                        - **Litigation Warning:** “Absent timely action, we are prepared to file a Writ of Mandamus in the U.S. District Court for the District of **[District]**, seeking to compel adjudication and recover EAJA fees and costs.”  
                        - **Attorney Availability:** “**[Attorney Name]** is available to supply any additional information or documentation USCIS may require.”  

                        **Very truly yours,**  
                        \_\_\_\_\_\_\_\_\_\_\_,  
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
            name="Assessment Report Agent", 
            template=(
                rf"""

                Today’s date is {current_date}.
                You are tasked with generating a Assessment Report for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                # EB‑1A I‑140 Support Letter – Extraordinary Ability Petition for [Beneficiary’s Full Name]  
                **Self‑Petitioner / Sponsor: [Beneficiary’s Full Name]**  
                **[Petitioner’s Name if Sponsor Organization]**  
                **[Petitioner’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** Assessment Report for EB‑1A I‑140 Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Petitioner:** “[Beneficiary’s Full Name], acting as self‑petitioner (the ‘Petitioner’), hereby submits this Assessment Report in support of Form I‑140 under EB‑1A classification for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                **Purpose:** To present a detailed evaluation of extraordinary ability in the sciences/business/arts and demonstrate eligibility under INA § 203(b)(1)(A) via evidence of sustained national or international acclaim.  
                **Summary:** This dossier outlines the Beneficiary’s achievements aligned to at least three of the ten regulatory EB‑1A criteria, followed by a totality‑of‑evidence merits discussion.

                **Overview of Extraordinary Ability Profile**  
                **Professional Summary:**  
                Education, awards, high‑impact contributions, media recognition, expert recommendation letters—framed to satisfy criteria under 8 CFR 204.5(h).  
                Key professional milestones, original contributions of major significance, roles as judge or reviewer, evidence of membership in elite associations, leading roles in distinguished organizations, and enhanced remuneration (if applicable).

                **Self‑Petition & Sponsorship**  
                As permitted under EB‑1A, no employer job offer or Labor Certification is required. The Petitioner sponsors themselves, demonstrating continuing intent to work in their field of expertise in the U.S., supported by contracts, client engagements, or project plans.

                **Summary of Qualifications**  
                **Educational Background:**  
                Degree(s) (e.g. Ph.D., M.S., B.S.) in **[Field]** from **[Institution]** (Date), including honors and recognition.  
                **Professional Achievements & Honors:**  
                Major awards, recognition by professional bodies, invited presentations, elected positions, or patents/publications establishing industry leadership.  
                **Criteria‑Specific Evidence:**  
                Each qualifying criterion is substantiated with labeled exhibits: publications about the individual, judging panels, membership proofs, expert support letters, citation indices, and project leadership documentation (see Exhibit List).

                **Regulatory Criteria Analysis**  
                — **Part One: Initial Criteria (8 CFR 204.5(h))**:  
                At least three criteria are met, including but not limited to:  
                • Evidence of original contributions of major significance  
                • Authorship of scholarly articles in major media  
                • Performance in a leading or critical role in distinguished organizations  
                • Judging the work of others  
                • Membership in associations requiring outstanding achievements  
                — **Part Two: Final Merits Determination (Kazarian standard)**:  
                Review of the totality of evidence supports that the Beneficiary is among the very small percentage at the top of the field and has sustained acclaim.

                **Supporting Documentation & Organizational Compliance**  
                — **Form I‑140:** Completed edition (mm/dd/yy) submitted with all required sections.  
                — **Form I‑907 (if premium processing requested):** Submitted concurrently.  
                — **Recommendation Letters:** Independent, expert support letters detailing the Beneficiary’s exceptional achievements and field contributions .  
                — **Publications, Media, Awards:** Packaged with translated materials, highlighted citations, and tabs corresponding to relevant criteria .  
                — **Exhibit Index:** Tabbed and cross‑referenced to eligibility criteria and items in Exhibit List.

                **Conclusion & Request**  
                **Eligibility Confirmed:** “Based on the foregoing, the Beneficiary clearly satisfies the extraordinary ability standards under INA § 203(b)(1)(A), meeting multiple regulatory criteria with evidence of sustained national/international acclaim.”  
                **Approval Requested:** “The Petitioner respectfully requests prompt approval of the EB‑1A I‑140 petition for **[Beneficiary’s Name]**, and stands ready to respond to any further inquiries.”  
                **Point of Contact:** “Please direct any questions to **[Petitioner’s Representative Name]**, **[Title]**, at **[Phone]** or **[Email]**.”  

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Authorized Signatory’s Name], [Title]**  
                **[Company or Self‑Petitioner Name]**
                

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
            name="Visa Application Summary Report Agent",
            template=(
                rf"""
                Today’s date is {current_date}.
                You are tasked with generating a Assessment Report for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                # EB‑1A I‑140 Application Summary Report – [Beneficiary’s Full Name]  
                **[Sponsor or Petitioner Name]**  
                **[Sponsor Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** EB‑1A I‑140 Petition Summary Report – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Sponsor’s Name] (the “Petitioner”) submits this I‑140 petition under the EB‑1A (Alien of Extraordinary Ability) category on behalf of **[Beneficiary’s Full Name]** (the “Beneficiary”).”  
                **Category & Intent:** EB‑1A classification as an individual of extraordinary ability under INA § 203(b)(1)(A). The Beneficiary self‑petitioned or is sponsored to demonstrate sustained national or international acclaim and intent to continue work in the field of extraordinary ability.  

                **Summary of Evidence**  
                The evidence clearly establishes that the Beneficiary meets the EB‑1A regulatory standard. Academic records and credential evaluations confirm advanced degree(s) in _[Field]_, aligning directly with the field of expertise. Documentation satisfies at least three of the ten regulatory criteria under 8 CFR 204.5(h), such as awards of national/international significance, original contributions to the field, authorship in high‑impact publications, judging peer work, and leadership roles in distinguished organizations. Extensive recommendation letters, citation metrics, publication indexes, and media references corroborate acclaim. Expert testimonials highlight influence and impact. No labor certification is required or submitted in this category.

                **Supporting Documents & Forms Enclosed**  
                - Form I‑140, properly completed and signed by Petitioner  
                - Form G‑28, Notice of Entry of Appearance as Attorney (if applicable)  
                - I‑797 receipt notice from any prior I‑140 approval (if beneficiary benefiting from earlier priority date)  
                - Detailed exhibit list mapping evidence to each regulatory criterion satisfied  
                - Academic diplomas, transcripts, and credential evaluation reports  
                - Expert recommendation letters, publication lists, citation reports  
                - Evidence of awards, membership in associations requiring excellence, media coverage, judging records, salary or remuneration comparisons  
                - Supplemental original contributions and organizational leadership documentation  

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the totality of evidence, **[Beneficiary’s Full Name]** unquestionably meets the statutory and regulatory standard for classification as an Alien of Extraordinary Ability under EB‑1A.”  
                **Request for Approval:** “Petitioner respectfully requests that USCIS adjudicate Form I‑140 under the EB‑1A category promptly and grant the Immigrant Petition without delay.”  
                **Point of Contact:** “For any questions or requests for additional documentation, please contact **[Petitioner’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Authorized Signatory Name], [Title]**  
                **[Sponsor or Petitioner Company or Firm]**  

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
    "I-140 Cover Letter": [
        # Header block
        "beneficiary full name",
        "sponsor name employer petitioner",
        "EB-1A extraordinary ability",
        "I-140 petition",
        "beneficiary country of citizenship nationality",
        
        # Beneficiary identity & pronoun
        "beneficiary gender pronoun he she they",
        "citizenship nationality passport",
        
        # Sponsor information
        "sponsor company name employer",
        "sponsor relationship to beneficiary",
        "job offer from sponsor",
        "sponsor support letter",
        
        # Field of expertise & classification
        "field of expertise area of extraordinary ability",
        "sustained national or international acclaim",
        "one of the small percentage at the very top of the field",
        
        # Regulatory criteria
        "awards prizes recognition",
        "membership associations outstanding achievements",
        "published material major media",
        "original contributions major significance",
        "scholarly articles publications citations",
        "leading or critical role distinguished organizations",
        "high salary remuneration",
        "commercial success",
        "judging the work of others",
        
        # Exhibits
        "submitted exhibits list supporting documents",
        
        # Signature block
        "attorney preparer name law firm name",
    ],
    "Support Letter": [
        # Document filenames
        "sponsor support letter",
        "expert recommendation letters",
        "letters of recommendation expert opinion",
        "resume curriculum vitae CV",
        "degree certificate diploma transcript",
        
        # Sponsor & beneficiary
        "sponsor company name employer",
        "beneficiary full name",
        "relationship between sponsor and beneficiary",
        "job title position offered by sponsor",
        
        # Field of expertise
        "field of expertise area of extraordinary ability",
        "sustained national or international acclaim",
        
        # Criteria evidence
        "awards prizes recognition",
        "membership associations",
        "published material major media",
        "original contributions major significance",
        "scholarly articles citations",
        "leading or critical role",
        "high salary remuneration",
        "commercial success",
        "judging the work of others",
        
        # Professional background
        "professional experience career summary",
        "key areas of expertise",
        "publications citations patents awards",
        
        # Education
        "degree major field of study university",
        "transcript educational credentials",
        
        # Future work
        "continued work in the field",
        "work will benefit the United States",
    ],
    "Intent to Depart": [
        # Header block
        "beneficiary full name",
        "sponsor name employer",
        "EB-1A extraordinary ability",
        "I-140 petition",
        "beneficiary country of citizenship nationality",
        
        # Identity & country
        "citizenship nationality country",
        "passport biographical pages passport copy",
        
        # Field & future work
        "field of expertise area of extraordinary ability",
        "intend to continue work in the area of extraordinary ability",
        "prospective work United States",
        "work will benefit the United States",
        
        # Sponsor-related
        "job offer from sponsor",
        "position offered by sponsor",
        "sponsor support",
        
        # Credentials
        "degree certificate diploma transcript resume",
        "beneficiary qualifications professional background",
        "publications citations awards patents",
        "letters of recommendation expert opinion",
        
        # Residence / ties
        "foreign home address permanent residence",
        "family ties property ownership",
    ],
    "Recommendation-Letter": [
        "recommendation letter",
        "expert letter",
        "EB-1A",
        "extraordinary ability",
        "original contributions",
        "major significance",
        "critical role",
        "awards-recognition",
        "publications",
        "media-reports",
        "membership-in-org",
    ],
    "Exhibit List": [
        "exhibit list",
        "EB-1A",
        "i-140",
        "g-1145",
        "g-28-company",
        "i-907",
        "all-degree-certs",
        "awards-recognition",
        "degree-evidence",
        "birth-certificate",
        "form-i-94",
        "form-w2-1099",
        "publications",
        "membership-in-org",
        "passport",
        "visa-pages",
        "recommendation-letters",
        "formation-documents",
        "federal-tax-returns",
        "media-reports",
    ],
    "Evidence-Summary Chart": [
        "evidence summary chart",
        "EB-1A criteria",
        "extraordinary ability",
        "awards-recognition",
        "publications",
        "membership-in-org",
        "media-reports",
        "recommendation-letters",
        "all-degree-certs",
        "degree-evidence",
        "form-w2-1099",
        "federal-tax-returns",
    ],
    "RFE Response Brief": [
        "request for evidence response",
        "RFE",
        "EB-1A",
        "I-140",
        "USCIS",
        "extraordinary ability",
        "sustained acclaim",
        "major significance",
        "awards-recognition",
        "publications",
        "media-reports",
        "recommendation-letters",
    ],
    "Demand Letter": [
        "demand letter",
        "pending adjudication",
        "adjudication delay",
        "EB-1A",
        "I-140",
        "USCIS",
        "i-907",
        "g-28-company",
        "filing receipt",
        "final decision",
    ],
    "Assessment Report": [
        "assessment report",
        "EB-1A",
        "I-140",
        "extraordinary ability",
        "sustained acclaim",
        "EB-1A criteria",
        "all-degree-certs",
        "awards-recognition",
        "degree-evidence",
        "publications",
        "membership-in-org",
        "media-reports",
        "recommendation-letters",
    ],
    "Visa Application Summary Report": [
        "visa application summary report",
        "EB-1A",
        "I-140",
        "i-140",
        "g-1145",
        "g-28-company",
        "i-907",
        "birth-certificate",
        "form-i-94",
        "passport",
        "visa-pages",
        "all-degree-certs",
        "recommendation-letters",
        "media-reports",
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