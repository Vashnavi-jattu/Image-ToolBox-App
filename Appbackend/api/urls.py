from django.urls import path
from api import views
from .views import hello_api

urlpatterns = [
    path('api/hello/', hello_api),  # This should match the Axios call
    path("control_tuning_imagebased/", views.control_tuning_imagebased),

    path('upload-image/', views.upload_image),
    path('extract_data/', views.extract_data),
    path('identify_foptd/', views.identify_foptd),
    path('identify_soptd/', views.identify_soptd),
    path('identify_integrator_delay/', views.identify_integrator_delay),
    path('identify_inverse_response_tf/', views.identify_inverse_response_tf),
    path('simulate_pid/', views.simulate_pid),
    path('simulate_pi/', views.simulate_pi),
    path('simulate_p/', views.simulate_p),
    path('simulate_close_loop_response/', views.simulate_close_loop_response),
    path('get_variables_report/', views.get_variables_report),
    path('tuning/', views.tuning, name='tuning'),
]
