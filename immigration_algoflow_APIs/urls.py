from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/aap/', include('aap.urls')),
    path('api/aea/', include('aea.urls')),
    path('api/ds_160/', include('ds_160.urls')),
    path('api/ds_260/', include('ds_260.urls')),
    path('api/eb-1aA/', include('eb_1aA.urls')),
    path('api/eb-1aB/', include('eb_1aB.urls')),
    path('api/naturalization/', include('naturalization.urls')),
    path('api/reentry_permit/', include('reentry_permit.urls')),
]

# This only runs when DEBUG=True
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)