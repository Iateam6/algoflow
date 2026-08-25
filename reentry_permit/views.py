from case_jobs.api.legacy_views import legacy_generate_doc, legacy_index
from case_jobs.api.views import (
    create_generation as shared_create_generation,
    download_generation as shared_download_generation,
)
from django.views.decorators.csrf import csrf_exempt

from .config import VISA_TYPE


def index(request):
    return legacy_index(request, VISA_TYPE)


@csrf_exempt
def generate_doc(request):
    return legacy_generate_doc(request, VISA_TYPE)


@csrf_exempt
def create_generation(request):
    return shared_create_generation(request, VISA_TYPE)


@csrf_exempt
def download_generation(request, job_id):
    return shared_download_generation(request, VISA_TYPE, str(job_id))
