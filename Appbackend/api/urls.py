# urls.py
from django.urls import path
from .views import hello_api
from . import views
from .views import upload_image
from .views import identify_foptd
from  .views import identify_soptd
from .views import identify_integrator_delay
from .views import identify_inverse_response_tf
from .views import simulate_pid
from .views import simulate_pi
from .views import simulate_p
from .views import simulate_close_loop_response
from .views import get_variables_report
from .views import tuning

#from api.views import GeneratePDF


urlpatterns = [
     # This should match the Axios call
    path('api/upload/', upload_image, name='upload_image'),
    path('api/identify_foptd/', identify_foptd),
    path('api/identify_soptd/', identify_soptd),
    path('api/identify_integrator_delay/', identify_integrator_delay),
    path('api/simulate_close_loop_response/', simulate_close_loop_response),
    path('api/get_variables_report/', get_variables_report),
    #path('api/generate-pdf/', GeneratePDF.as_view(), name='generate-pdf'),
    # the follwoing are not used in the app
    
]
