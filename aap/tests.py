from django.test import SimpleTestCase

from case_jobs.tests.visa_contract import VisaPackageContractMixin


class TNContractTests(VisaPackageContractMixin, SimpleTestCase):
    module_name = "tn"
    url_prefix = "/api/tn/"
