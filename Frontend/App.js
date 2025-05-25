import React, { useState, useRef } from 'react';
import { 
  View, 
  Text, 
  TextInput,
  StyleSheet, 
  TouchableOpacity, 
  Platform,
  ActivityIndicator,
  ScrollView,
  SafeAreaView,
  Image,
  PermissionsAndroid, 
  Alert,
  Dimensions,
  KeyboardAvoidingView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import Icon from 'react-native-vector-icons/FontAwesome';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import { WebView } from 'react-native-webview';
//import ViewShot from 'react-native-view-shot';import * as Print from 'expo-print';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system';

const { width } = Dimensions.get('window');
const API_URL = 'http://192.168.1.162';

// Step Indicator Component
const StepIndicator = ({ currentStep, totalSteps }) => {
  return (
    <View style={styles.stepIndicatorContainer}>
      {Array.from({ length: totalSteps }).map((_, index) => (
        <React.Fragment key={index}>
          <View style={[
            styles.stepCircle,
            currentStep > index + 1 && styles.completedStep,
            currentStep === index + 1 && styles.activeStep
          ]}>
            {currentStep > index + 1 ? (
              <Icon name="check" size={12} color="white" />
            ) : (
              <Text style={[
                styles.stepNumber,
                currentStep === index + 1 && styles.activeStepNumber
              ]}>{index + 1}</Text>
            )}
          </View>
          {index < totalSteps - 1 && (
            <View style={[
              styles.stepLine,
              currentStep > index + 1 && styles.completedLine
            ]} />
          )}
        </React.Fragment>
      ))}
      <Text style={styles.stepText}>
        Step {currentStep} of {totalSteps}
      </Text>
    </View>
  );
};

// Home Screen (Original Uploader Screen)
function ImageUploaderScreen({ navigation }) {
  const [imageUri, setImageUri] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [inputs, setInputs] = useState({
    y_min: '0',
    y_max: '100',
    x_min: '0',
    x_max: '100',
    op_color: 'red',
    pv_color: 'blue'
  });
  const [showDropdown, setShowDropdown] = useState({
    op: false,
    pv: false
  });

  const handleInputChange = (name, value) => {
    setInputs(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const toggleDropdown = (type) => {
    setShowDropdown(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const handleColorSelect = (type, value) => {
    handleInputChange(type, value);
    setShowDropdown(prev => ({
      ...prev,
      [type.replace('_color', '')]: false
    }));
  };

  const colorOptions = [
    { label: 'Red', value: 'red' },
    { label: 'Blue', value: 'blue' },
    { label: 'Green', value: 'green' },
    { label: 'Purple', value: 'purple' },
    { label: 'Orange', value: 'orange' },
    { label: 'Black', value: 'black' }
  ];

  const pickImage = async (source) => {
    let result;
    
    if (source === 'gallery') {
      result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });
    } else {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Sorry, we need camera permissions to make this work!');
        return;
      }
      
      result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });
    }

    if (!result.canceled && result.assets) {
      setImageUri(result.assets[0].uri);
      setUploadStatus('');
    }
  };

  const uploadImage = async () => {
    if (!imageUri) {
      Alert.alert('Error', 'Please select an image first');
      return;
    }

    setIsLoading(true);
    setUploadStatus('Uploading...');
    
    try {
      const formData = new FormData();
      formData.append('image', {
        uri: Platform.OS === 'android' ? imageUri.replace('file://', '') : imageUri,
        name: 'upload.jpg',
        type: 'image/jpeg'
      });
      
      formData.append('params', JSON.stringify({
        y_min: parseFloat(inputs.y_min),
        y_max: parseFloat(inputs.y_max),
        x_min: parseFloat(inputs.x_min),
        x_max: parseFloat(inputs.x_max),
        op_color: inputs.op_color,
        pv_color: inputs.pv_color
      }));

      const response = await axios.post(
        `${API_URL}:8000/api/upload/`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setUploadStatus('Upload successful!');
      
      navigation.navigate('Step2', {
        originalImage: imageUri,
        graphHtml: response.data.graph_html,
        parameters: inputs
      });
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.flexOne}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 0}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContainer}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.container}>
            <StepIndicator currentStep={1} totalSteps={5} />
          
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Image Upload</Text>
            </View>
            <View style={styles.cardBody}>
              <View style={styles.imageButtonsContainer}>
                <TouchableOpacity 
                  style={[styles.imageButton, styles.galleryButton, isLoading && styles.disabledButton]}
                  onPress={() => pickImage('gallery')}
                  disabled={isLoading}
                >
                  <Icon name="photo" size={20} color="white" />
                  <Text style={styles.buttonText}> Gallery</Text>
                </TouchableOpacity>
                
                <TouchableOpacity 
                  style={[styles.imageButton, styles.cameraButton, isLoading && styles.disabledButton]}
                  onPress={() => pickImage('camera')}
                  disabled={isLoading}
                >
                  <Icon name="camera" size={20} color="white" />
                  <Text style={styles.buttonText}> Camera</Text>
                </TouchableOpacity>
              </View>

              {imageUri && (
                <Image source={{ uri: imageUri }} style={styles.imagePreview} />
              )}

              <View style={styles.inputSection}>
                <Text style={styles.sectionTitle}>Image Parameters</Text>
                
                <View style={styles.inputRow}>
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>X min</Text>
                    <TextInput
                      style={styles.input}
                      value={inputs.x_min}
                      onChangeText={(text) => handleInputChange('x_min', text)}
                      keyboardType="numeric"
                      placeholder="0"
                    />
                  </View>
                  
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>X max</Text>
                    <TextInput
                      style={styles.input}
                      value={inputs.x_max}
                      onChangeText={(text) => handleInputChange('x_max', text)}
                      keyboardType="numeric"
                      placeholder="0"
                    />
                  </View>
                </View>

                <View style={styles.inputRow}>
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>Y min</Text>
                    <TextInput
                      style={styles.input}
                      value={inputs.y_min}
                      onChangeText={(text) => handleInputChange('y_min', text)}
                      keyboardType="numeric"
                      placeholder="0"
                    />
                  </View>
                  
                  <View style={styles.inputGroup}>
                    <Text style={styles.inputLabel}>Y max</Text>
                    <TextInput
                      style={styles.input}
                      value={inputs.y_max}
                      onChangeText={(text) => handleInputChange('y_max', text)}
                      keyboardType="numeric"
                      placeholder="0"
                    />
                  </View>
                </View>

                <View style={styles.colorPickerRow}>
                  <View style={styles.colorPickerGroup}>
                    <Text style={styles.inputLabel}>OP Color</Text>
                    <TouchableOpacity 
                      style={styles.colorPickerButton}
                      onPress={() => toggleDropdown('op')}
                    >
                      <Text style={styles.colorPickerText}>{inputs.op_color}</Text>
                      <Icon 
                        name={showDropdown.op ? "chevron-up" : "chevron-down"} 
                        size={16} 
                        color="#333" 
                      />
                    </TouchableOpacity>
                    {showDropdown.op && (
                      <View style={styles.colorDropdown}>
                        {colorOptions.map(option => (
                          <TouchableOpacity
                            key={`op_${option.value}`}
                            style={styles.colorOption}
                            onPress={() => handleColorSelect('op_color', option.value)}
                          >
                            <Text>{option.label}</Text>
                          </TouchableOpacity>
                        ))}
                      </View>
                    )}
                  </View>

                  <View style={styles.colorPickerGroup}>
                    <Text style={styles.inputLabel}>PV Color</Text>
                    <TouchableOpacity 
                      style={styles.colorPickerButton}
                      onPress={() => toggleDropdown('pv')}
                    >
                      <Text style={styles.colorPickerText}>{inputs.pv_color}</Text>
                      <Icon 
                        name={showDropdown.pv ? "chevron-up" : "chevron-down"} 
                        size={16} 
                        color="#333" 
                      />
                    </TouchableOpacity>
                    {showDropdown.pv && (
                      <View style={styles.colorDropdown}>
                        {colorOptions.map(option => (
                          <TouchableOpacity
                            key={`pv_${option.value}`}
                            style={styles.colorOption}
                            onPress={() => handleColorSelect('pv_color', option.value)}
                          >
                            <Text>{option.label}</Text>
                          </TouchableOpacity>
                        ))}
                      </View>
                    )}
                  </View>
                </View>
              </View>

              <TouchableOpacity 
                style={[styles.primaryButton, (!imageUri || isLoading) && styles.disabledButton]}
                onPress={uploadImage}
                disabled={!imageUri || isLoading}
              >
                {isLoading ? (
                  <ActivityIndicator size="small" color="white" />
                ) : (
                  <Text style={styles.buttonText}>Upload Image</Text>
                )}
              </TouchableOpacity>

              {uploadStatus ? <Text style={styles.statusText}>{uploadStatus}</Text> : null}
            </View>
          </View>
        </View>
      </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// Results Screen
function Step2({ route, navigation }) {
  
  // Get data from previous step
  const { originalImage } = route.params || {};
  
  // Your component state
  const [processedImage, setProcessedImage] = React.useState(null);
  const [step2GraphHtml, setStep2GraphHtml] = React.useState('');
  const [x_min, setXMin] = React.useState(0);
  const [x_max, setXMax] = React.useState(1);
  const [y_min, setYMin] = React.useState(0);
  const [y_max, setYMax] = React.useState(1);

  const { graphHtml, parameters } = route.params;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.container}>
          <StepIndicator currentStep={2} totalSteps={5} />
          
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Extraction Results</Text>
            </View>
            <View style={styles.cardBody}>
              {/* Extracted Graph */}
              <View style={styles.resultSection}>
                <Text style={styles.sectionTitle}>Extracted Data Graph</Text>
                <WebView
                  originWhitelist={['*']}
                  source={{ html: graphHtml }}
                  style={styles.webview}
                />
              </View>

              {/* Parameters Card */}
              <View style={styles.parametersSection}>
                <Text style={styles.sectionTitle}>Used Parameters</Text>
                
                <View style={styles.parametersGrid}>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>X min:</Text>
                    <Text style={styles.parameterValue}>{parameters.x_min}</Text>
                  </View>
                  
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>X max:</Text>
                    <Text style={styles.parameterValue}>{parameters.x_max}</Text>
                  </View>

                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>Y min:</Text>
                    <Text style={styles.parameterValue}>{parameters.y_min}</Text>
                  </View>
                  
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>Y max:</Text>
                    <Text style={styles.parameterValue}>{parameters.y_max}</Text>
                  </View>

                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>OP Color:</Text>
                    <Text style={styles.parameterValue}>{parameters.op_color}</Text>
                  </View>
                  
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>PV Color:</Text>
                    <Text style={styles.parameterValue}>{parameters.pv_color}</Text>
                  </View>
                </View>
              </View>

              <View style={styles.buttonContainer}>
                <TouchableOpacity 
                  style={styles.primaryButton}
                  onPress={() => navigation.navigate('Step3', {
                    ...route.params, // Keep original image data
                    graphHtml,   
                    parameters: {           // Your extraction parameters
                      x_min,
                      x_max,
                      y_min,
                      y_max,
                      op_color: parameters.op_color,
                      pv_color: parameters.pv_color
                    },
                    step2GraphHtml          // Graph HTML from Step2
                  })}
                >
                  <Text style={styles.buttonText}>NEXT</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const Step3 = ({ navigation, route }) => {
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedModel, setSelectedModel] = useState('Select Model');
  const [isLoading, setIsLoading] = useState(false);
  //const [modelResults, setModelResults] = useState(null);
  const [graphHtml, setGraphHtml] = useState('');
  const [error, setError] = useState(null);
  const [modelResults, setModelResults] = useState({});

  const params = route.params || {};
  const { parameters = {}, graphData = '' } = params;

  const handleModelSelect = async (model) => {
    setSelectedModel(model);
    setShowDropdown(false);
    setIsLoading(true);
    setError(null);
    
    try {
      let endpoint = '';
      
      if (model === 'FOPTD Identification') {
        endpoint = 'identify_foptd';
      } else if (model === 'SOPTD Identification') {
        endpoint = 'identify_soptd';
      } else if (model === 'Integrator Plus Dead Time (IPDT)') {
        endpoint = 'identify_integrator_delay';
      }

      const response = await axios.post(
        `${API_URL}:8000/api/${endpoint}/`,
        {
          parameters: parameters,
          graph_data: graphData
        },
        {
          timeout: 30000
        }
      );
     
      if (!response.data) {
        throw new Error('No data received from server');
      }

      setModelResults(response.data);
      setGraphHtml(response.data.modeling_graph_html || '');
      
    } catch (error) {
      console.error('Model identification error:', error);
      setError(error.message || 'Failed to perform model identification');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to render fitted parameters based on model type
  const renderFittedParameters = () => {
 if (!modelResults) return null;

  // Extract parameters from different possible response structures
  const params = modelResults.parameters || 
                modelResults.modeling_params || 
                {
                  Kp: modelResults.Kp_fit || modelResults.Kp_est,
                  tau: modelResults.tau_fit || modelResults.tau_est,
                  zeta: modelResults.zeta_est,
                  theta: modelResults.theta_fit || modelResults.theta_est
                };

  // Format value with consistent decimal places
  const formatValue = (value) => {
    if (value === undefined || value === null) return 'N/A';
    if (typeof value === 'number') return value.toFixed(2);
    return value;
  };


  if (selectedModel === 'FOPTD Identification') {
    return (
      <View style={styles.fittedParamsContainer}>
        <Text style={styles.sectionSubtitle}>Fitted Parameters:</Text>
        <View style={styles.parametersContainer}>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>Kp (Gain):</Text>
            <Text style={styles.parameterValue}>
              {params[0]?.toFixed(4) || 'N/A'}
            </Text>
          </View>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>τ (Time Constant):</Text>
            <Text style={styles.parameterValue}>
              {params[1]?.toFixed(4) || 'N/A'}
            </Text>
          </View>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>θ (Time Delay):</Text>
            <Text style={styles.parameterValue}>
              {params[2]?.toFixed(4) || 'N/A'}
            </Text>
          </View>
        </View>
      </View>
    );
  } else if (selectedModel === 'SOPTD Identification') {
    return (
      <View style={styles.fittedParamsContainer}>
        <Text style={styles.sectionSubtitle}>Fitted Parameters:</Text>
        <View style={styles.parametersContainer}>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>Kp (Gain):</Text>
            <Text style={styles.parameterValue}>
              {formatValue(params.Kp)}
            </Text>
          </View>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>τ (Time Constant):</Text>
            <Text style={styles.parameterValue}>
              {formatValue(params.tau)}
            </Text>
          </View>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>ζ (Damping Ratio):</Text>
            <Text style={styles.parameterValue}>
              {formatValue(params.zeta)}
            </Text>
          </View>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>θ (Time Delay):</Text>
            <Text style={styles.parameterValue}>
              {formatValue(params.theta)}
            </Text>
          </View>
        </View>
      </View>
    );
  
  } else if (selectedModel === 'Integrator Plus Dead Time (IPDT)') {
    return (
      <View style={styles.fittedParamsContainer}>
        <Text style={styles.sectionSubtitle}>Fitted Parameters:</Text>
        <View style={styles.parametersContainer}>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>Kp (Gain):</Text>
            <Text style={styles.parameterValue}>
              {params[0]?.toFixed(4) || 'N/A'}
            </Text>
          </View>
          <View style={styles.parameterRow}>
            <Text style={styles.parameterLabel}>θ (Time Delay):</Text>
            <Text style={styles.parameterValue}>
              {params[1]?.toFixed(4) || 'N/A'}
            </Text>
          </View>
        </View>
      </View>
    );
  }
  return null;
};

  return (
  <SafeAreaView style={styles.safeArea}>
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        <StepIndicator currentStep={3} totalSteps={5} />
        
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>System Identification</Text>
          </View>
          <View style={styles.cardBody}>
            {error && (
              <View style={styles.errorContainer}>
                <Text style={styles.errorText}>{error}</Text>
              </View>
            )}

            <View style={styles.dropdownContainer}>
              <TouchableOpacity 
                style={styles.dropdownButton}
                onPress={() => setShowDropdown(!showDropdown)}
                disabled={isLoading}
              >
                <Text style={styles.dropdownButtonText}>{selectedModel}</Text>
                <Icon 
                  name={showDropdown ? "chevron-up" : "chevron-down"} 
                  size={20} 
                  color="#fff" 
                />
              </TouchableOpacity>

              {showDropdown && (
                <View style={styles.dropdownMenu}>
                  <TouchableOpacity 
                    style={styles.dropdownItem}
                    onPress={() => handleModelSelect('FOPTD Identification')}
                  >
                    <Text>FOPTD Identification</Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={styles.dropdownItem}
                    onPress={() => handleModelSelect('SOPTD Identification')}
                  >
                    <Text>SOPTD Identification</Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={styles.dropdownItem}
                    onPress={() => handleModelSelect('Integrator Plus Dead Time (IPDT)')}
                  >
                    <Text>Integrator Plus Dead Time (IPDT)</Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>

            {isLoading && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#3498db" />
                <Text style={styles.loadingText}>Identifying model parameters...</Text>
              </View>
            )}

            {/* Results Section - This was missing */}
            {Object.keys(modelResults).length > 0 && (
              <View style={styles.resultsContainer}>
                <Text style={styles.resultsTitle}>{selectedModel} Results</Text>
                
                {/* Graph Display */}
                {graphHtml ? (
                  <WebView
                    originWhitelist={['*']}
                    source={{ html: graphHtml }}
                    style={styles.modelGraph}
                  />
                ) : (
                  <Text style={styles.noGraphText}>No graph data available</Text>
                )}

                {/* Parameters Display */}
                {renderFittedParameters()}
              </View>
            )}
          </View>
        </View>

        <View style={styles.navButtonsContainer}>
          <TouchableOpacity 
            style={styles.prevButton} 
            onPress={() => navigation.goBack()}
            disabled={isLoading}
          >
            <Text style={styles.buttonText}>PREVIOUS</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.nextButton, Object.keys(modelResults).length === 0 && styles.disabledButton]} 
            onPress={() => navigation.navigate('Step4', {
              ...route.params,
              modelType: selectedModel,
              modelParams: modelResults.modeling_params || [],
              modelParamNames: modelResults.modeling_params_names || [],
              modelingGraphHtml: graphHtml
            })}
            disabled={Object.keys(modelResults).length === 0 || isLoading}
          >
            <Text style={styles.buttonText}>NEXT</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  </SafeAreaView>
);
};


const Step4 = ({ navigation, route }) => {
  const { modelParams, modelType, graphHtml: initialGraphHtml } = route.params || {};
  const [showCriteriaDropdown, setShowCriteriaDropdown] = useState(false);
  const [selectedCriteria, setSelectedCriteria] = useState('Select Criteria');
  const [selectedController, setSelectedController] = useState('');
  const [activeSubmenu, setActiveSubmenu] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [tuningResults, setTuningResults] = useState(null);
  const [graphHtml, setGraphHtml] = useState(initialGraphHtml || '');
  const [error, setError] = useState(null);
   const [tuningHistory, setTuningHistory] = useState([]); // Store all tuning results

  const criteriaOptions = [
    { name: 'ISE', controllers: ['PID', 'PI', 'P'] },
    { name: 'IAE', controllers: ['PID', 'PI', 'P'] },
    { name: 'ITAE', controllers: ['PID', 'PI', 'P'] },
    { name: 'Inverse Response', controllers: ['Identify'] }

  ];

  // Model type mapping
  const modelTypeMapping = {
    'FOPTD Identification': 'First-Order Plus Dead Time (FOPDT)',
    'SOPTD Identification': 'Second-Order Plus Dead Time (SOPDT)',
    'Integrator Plus Dead Time (IPDT)': 'Integrator Plus Dead Time (IPDT)'
  };

  const handleCriteriaSelect = (criteria) => {
    setSelectedCriteria(criteria);
    setActiveSubmenu(activeSubmenu === criteria ? null : criteria);
  };

  const handleControllerSelect = async (criteria, controller) => {
    setSelectedController(`${criteria}_${controller}`);
    setShowCriteriaDropdown(false);
    setActiveSubmenu(null);
    setIsLoading(true);
    setError(null);

    try {
      let endpoint = '';
      let payload = new FormData();  // Use FormData instead of raw JSON

      if (criteria === 'Inverse Response' && controller === 'Identify') {
        endpoint = 'identify_inverse_response_tf';
      } else {
        endpoint = 'simulate_close_loop_response';
        payload.append('criteria', criteria);
        payload.append('controller_type', controller);
        payload.append('modeling_type', modelType); // Make sure modelType is defined
        
        // Include modeling parameters if available
        if (modelParams) {
          payload.append('modeling_params', JSON.stringify(modelParams));
        }
      }
      
      console.log('Sending payload:', {
        criteria: criteria,
        controller_type: controller,
        modeling_type: modelType,
        modeling_params: modelParams
      });

      const response = await axios.post(
        `${API_URL}:8000/api/${endpoint}/`,
        payload,
        {
          headers: {
            'Content-Type': 'multipart/form-data',  // Change content type
          },
          timeout: 30000
        }
      );

      if (!response.data) {
        throw new Error('No data received from server');
      }
      const newTuningResult = {
      criteria,
      controller,
      parameters: response.data.parameters,
      parameter_names: response.data.parameter_names,
      graphHtml: response.data.modeling_graph_html || response.data.tuning_graph_html || ''
    };

     setTuningResults(newTuningResult);
     setGraphHtml(newTuningResult.graphHtml);
     setTuningHistory(prev => [...prev, newTuningResult]); // Add to history
      
    } catch (error) {
      console.error('Tuning error:', error);
      setError(error.response?.data?.error || error.message || 'Failed to perform controller tuning');
    } finally {
      setIsLoading(false);
    }
};

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.container}>
          <StepIndicator currentStep={4} totalSteps={5} />
          
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Controller Tuning</Text>
            </View>
            <View style={styles.cardBody}>
              {error && (
                <View style={styles.errorContainer}>
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}

              <View style={styles.dropdownContainer}>
                <TouchableOpacity 
                  style={styles.dropdownButton}
                  onPress={() => setShowCriteriaDropdown(!showCriteriaDropdown)}
                  disabled={isLoading}
                >
                  <Text style={styles.dropdownButtonText}>
                    {selectedController || selectedCriteria}
                  </Text>
                  <Icon 
                    name={showCriteriaDropdown ? "chevron-up" : "chevron-down"} 
                    size={20} 
                    color="#fff" 
                  />
                </TouchableOpacity>

                {showCriteriaDropdown && (
                  <View style={styles.dropdownMenu}>
                    {criteriaOptions.map((criteria) => (
                      <View key={criteria.name} style={styles.submenuContainer}>
                        <TouchableOpacity
                          style={styles.criteriaItem}
                          onPress={() => handleCriteriaSelect(criteria.name)}
                        >
                          <Text>{criteria.name}</Text>
                          <Icon 
                            name={activeSubmenu === criteria.name ? "chevron-up" : "chevron-down"} 
                            size={16} 
                            color="#333" 
                          />
                        </TouchableOpacity>
                        
                        {activeSubmenu === criteria.name && (
                          <View style={styles.submenu}>
                            {criteria.controllers.map((controller) => (
                              <TouchableOpacity
                                key={`${criteria.name}_${controller}`}
                                style={styles.submenuItem}
                                onPress={() => handleControllerSelect(criteria.name, controller)}
                              >
                                <Text>{controller}</Text>
                              </TouchableOpacity>
                            ))}
                          </View>
                        )}
                      </View>
                    ))}
                  </View>
                )}
              </View>

              {isLoading && (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator size="large" color="#3498db" />
                  <Text style={styles.loadingText}>Calculating optimal parameters...</Text>
                </View>
              )}

              {tuningResults && (
                <View style={styles.resultsContainer}>
                  <Text style={styles.resultsTitle}>
                    {selectedController.split('_')[0]} - {selectedController.split('_')[1]} Results
                  </Text>
                  
                  {graphHtml ? (
                    <WebView
                      originWhitelist={['*']}
                      source={{ html: graphHtml }}
                      style={styles.modelGraph}
                    />
                  ) : (
                    <Text style={styles.noGraphText}>No tuning results available</Text>
                  )}

                  {tuningResults.parameters && (
                    <View style={styles.parametersContainer}>
                      {tuningResults.parameters.map((param, index) => (
                        <View key={index} style={styles.parameterRow}>
                          <Text style={styles.parameterLabel}>
                            {tuningResults.parameter_names?.[index] || `Parameter ${index + 1}`}:
                          </Text>
                          <Text style={styles.parameterValue}>{param.toFixed(2)}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}
            </View>
          </View>

          <View style={styles.navButtonsContainer}>
            <TouchableOpacity 
              style={styles.prevButton} 
              onPress={() => navigation.goBack()}
              disabled={isLoading}
            >
              <Text style={styles.buttonText}>PREVIOUS</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.nextButton, (!tuningResults || isLoading) && styles.disabledButton]} 
              
              // When navigating to Step5:
              onPress={() => navigation.navigate('Step5', {
              ...route.params, // All previous data
              controllerType: selectedController.split('_')[1],
              criteria: selectedController.split('_')[0],
              tuningParams: tuningResults?.parameters || [],
              tuningParamNames: tuningResults?.parameter_names || [],
              tuningGraphHtml: graphHtml,
              tuningHistory: tuningHistory // Pass the entire history
            })}
              disabled={!tuningResults || isLoading}
            >
              <Text style={styles.buttonText}>NEXT</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};
const Step5 = ({ navigation, route }) => {
  const {
    graphHtml = '',
    parameters = {
      x_min: 0,
      x_max: 0,
      y_min: 0,
      y_max: 0,
      op_color: '',
      pv_color: ''
    },
    modelType = '',
    modelParams = [],
    modelParamNames = [],
    modelingGraphHtml = '',
    controllerType = '',
    criteria = '',
    tuningParams = [],
    tuningParamNames = [],
    tuningGraphHtml = '',
    tuningHistory = [],
    theta_fit = 0,
    tau_fit = 0,
    Kp_fit = 0,
  } = route.params || {};

  const generateHTML = () => {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <style>
          body { font-family: Arial; padding: 20px; color: #333; background-color: #f5f5f5; }
          h1 { color: #2c3e50; text-align: center; }
          .container { padding: 10px; }
          .card { background-color: white; border-radius: 8px; margin-bottom: 20px; shadow-color: #000; shadow-offset: { width: 0, height: 2 }; shadow-opacity: 0.1; shadow-radius: 4; elevation: 3; }
          .card-header { background-color: #f8f9fa; padding: 15px; border-bottom-width: 1px; border-bottom-color: #ddd; border-top-left-radius: 8px; border-top-right-radius: 8px; }
          .card-title { font-size: 18px; font-weight: bold; }
          .card-body { padding: 15px; }
          .section-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #3498db; }
          .section-subtitle { font-size: 15px; font-weight: bold; margin-bottom: 8px; }
          .parameters-grid { flex-direction: row; flex-wrap: wrap; justify-content: space-between; margin-bottom: 15px; }
          .parameter-item { width: 48%; margin-bottom: 8px; }
          .parameter-label { font-weight: bold; color: #7f8c8d; }
          .parameter-value { color: #333; }
          .webview { height: 300px; width: 100%; background-color: transparent; }
          .modelGraph { height: 250px; width: 100%; background-color: transparent; }
          .graph-container { margin-vertical: 20px; }
          .graph-title { font-size: 14px; margin-bottom: 5px; text-align: center; }
          .parameters-container { margin-bottom: 15px; }
          .parameter-row { flex-direction: row; margin-bottom: 6px; }
          .no-data { color: #95a5a6; font-style: italic; }
          .tuningResultContainer { margin-bottom: 20px; }
          .footer { margin-top: 30px; text-align: center; color: #95a5a6; font-size: 12px; }
        </style>
      </head>
      <body>
        <h1>Process Control Report</h1>
        <div class="container">
          ${generateReportContent()}
          <div class="footer">
            Report generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}
          </div>
        </div>
      </body>
      </html>
    `;
  };

  const generateReportContent = () => {
    return `
      <!-- Extraction Results -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Extraction Results</div>
        </div>
        <div class="card-body">
          <div class="section-title">Extracted Data Graph</div>
          ${graphHtml || '<p class="no-data">No graph available</p>'}
          
          <div class="section-title">Used Parameters</div>
          <div class="parameters-grid">
            ${generateParameterItems()}
          </div>
        </div>
      </div>
      
      <!-- System Identification -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">System Identification</div>
        </div>
        <div class="card-body">
          <div class="section-subtitle">Model Type: ${modelType}</div>
          ${generateModelParameters()}
          ${modelingGraphHtml || '<p class="no-data">No modeling graph available</p>'}
        </div>
      </div>
      
      <!-- Controller Tuning -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Controller Tuning Results</div>
        </div>
        <div class="card-body">
          ${generateTuningHistory()}
        </div>
      </div>
    `;
  };

  const generateParameterItems = () => {
    return `
      <div class="parameter-item">
        <span class="parameter-label">X min:</span>
        <span class="parameter-value">${parameters.x_min}</span>
      </div>
      <div class="parameter-item">
        <span class="parameter-label">X max:</span>
        <span class="parameter-value">${parameters.x_max}</span>
      </div>
      <div class="parameter-item">
        <span class="parameter-label">Y min:</span>
        <span class="parameter-value">${parameters.y_min}</span>
      </div>
      <div class="parameter-item">
        <span class="parameter-label">Y max:</span>
        <span class="parameter-value">${parameters.y_max}</span>
      </div>
      <div class="parameter-item">
        <span class="parameter-label">OP Color:</span>
        <span class="parameter-value">${parameters.op_color}</span>
      </div>
      <div class="parameter-item">
        <span class="parameter-label">PV Color:</span>
        <span class="parameter-value">${parameters.pv_color}</span>
      </div>
    `;
  };

  const generateModelParameters = () => {
    return `
      <div class="parameters-container">
        <div class="parameter-row">
          <span class="parameter-label">Kp (Gain):</span>
          <span class="parameter-value">${Kp_fit.toFixed(4)}</span>
        </div>
        <div class="parameter-row">
          <span class="parameter-label">τ (Time Constant):</span>
          <span class="parameter-value">${tau_fit.toFixed(4)}</span>
        </div>
        <div class="parameter-row">
          <span class="parameter-label">θ (Time Delay):</span>
          <span class="parameter-value">${theta_fit.toFixed(4)}</span>
        </div>
        ${modelParams.map((param, index) => `
          <div class="parameter-row">
            <span class="parameter-label">${modelParamNames[index]}:</span>
            <span class="parameter-value">
              ${typeof param === 'number' ? param.toFixed(4) : param}
            </span>
          </div>
        `).join('')}
      </div>
    `;
  };

  const generateTuningHistory = () => {
    if (tuningHistory.length === 0) {
      return '<p class="no-data">No tuning results available</p>';
    }
    
    return tuningHistory.map((tuning, index) => `
      <div class="tuningResultContainer">
        <div class="section-subtitle">
          ${tuning.controller} Controller (${tuning.criteria} Criteria)
        </div>
        <div class="parameters-container">
          ${tuning.parameter_names.map((name, idx) => `
            <div class="parameter-row">
              <span class="parameter-label">${name}:</span>
              <span class="parameter-value">
                ${typeof tuning.parameters[idx] === 'number' 
                  ? tuning.parameters[idx].toFixed(4) 
                  : tuning.parameters[idx]}
              </span>
            </div>
          `).join('')}
        </div>
        ${tuning.graphHtml || '<p class="no-data">No tuning graph available</p>'}
      </div>
    `).join('');
  };

  const handleSavePDF = async () => {
    try {
      const html = generateHTML();
      const { uri } = await Print.printToFileAsync({
        html,
        width: 595,
        height: 842,
      });

      if (Platform.OS === 'android') {
        const permissions = await FileSystem.StorageAccessFramework.requestDirectoryPermissionsAsync();
        if (permissions.granted) {
          const newUri = await FileSystem.StorageAccessFramework.createFileAsync(
            permissions.directoryUri,
            `ProcessReport_${Date.now()}.pdf`,
            'application/pdf'
          );
          await FileSystem.writeAsStringAsync(newUri, await FileSystem.readAsStringAsync(uri), {
            encoding: FileSystem.EncodingType.Base64,
          });
          Alert.alert('Success', 'PDF saved to your Downloads folder');
          return;
        }
      }

      await Sharing.shareAsync(uri, {
        mimeType: 'application/pdf',
        dialogTitle: 'Save Process Report',
      });

    } catch (error) {
      Alert.alert('Error', 'Failed to generate PDF: ' + error.message);
      console.error(error);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.container}>
          <StepIndicator currentStep={5} totalSteps={5} />
          
          {/* EXTRACTION RESULTS */}
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Extraction Results</Text>
            </View>
            <View style={styles.cardBody}>
              <View style={styles.resultSection}>
                <Text style={styles.sectionTitle}>Extracted Data Graph</Text>
                <WebView
                  originWhitelist={['*']}
                  source={{ html: graphHtml }}
                  style={styles.webview}
                />
              </View>

              <View style={styles.parametersSection}>
                <Text style={styles.sectionTitle}>Used Parameters</Text>
                <View style={styles.parametersGrid}>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>X min:</Text>
                    <Text style={styles.parameterValue}>{parameters.x_min}</Text>
                  </View>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>X max:</Text>
                    <Text style={styles.parameterValue}>{parameters.x_max}</Text>
                  </View>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>Y min:</Text>
                    <Text style={styles.parameterValue}>{parameters.y_min}</Text>
                  </View>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>Y max:</Text>
                    <Text style={styles.parameterValue}>{parameters.y_max}</Text>
                  </View>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>OP Color:</Text>
                    <Text style={styles.parameterValue}>{parameters.op_color}</Text>
                  </View>
                  <View style={styles.parameterItem}>
                    <Text style={styles.parameterLabel}>PV Color:</Text>
                    <Text style={styles.parameterValue}>{parameters.pv_color}</Text>
                  </View>
                </View>
              </View>
            </View>
          </View>

          {/* SYSTEM IDENTIFICATION */}
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>System Identification</Text>
            </View>
            <View style={styles.cardBody}>
              <Text style={styles.sectionSubtitle}>Model Type: {modelType}</Text>
             

              {modelParams.length > 0 && (
                <View style={styles.parametersContainer}>
                  {modelParamNames.map((name, index) => (
                    <View key={index} style={styles.parameterRow}>
                      <Text style={styles.parameterLabel}>{name}:</Text>
                      <Text style={styles.parameterValue}>
                        {typeof modelParams[index] === 'number' 
                          ? modelParams[index].toFixed(4) 
                          : modelParams[index]}
                      </Text>
                    </View>
                  ))}
                </View>
              )}

              {modelingGraphHtml ? (
                <View style={styles.graphContainer}>
                  <Text style={styles.graphTitle}>Process Model Response</Text>
                  <WebView
                    originWhitelist={['*']}
                    source={{ html: modelingGraphHtml }}
                    style={styles.modelGraph}
                  />
                </View>
              ) : (
                <Text style={styles.noDataText}>No modeling graph available</Text>
              )}
            </View>
          </View>

          {/* CONTROLLER TUNING */}
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Controller Tuning Results</Text>
            </View>
            <View style={styles.cardBody}>
              {tuningHistory.length > 0 ? (
                tuningHistory.map((tuning, index) => (
                  <View key={index} style={styles.tuningResultContainer}>
                    <Text style={styles.sectionSubtitle}>
                      {tuning.controller} Controller ({tuning.criteria} Criteria)
                    </Text>
                    
                    {tuning.parameters && tuning.parameters.length > 0 && (
                      <View style={styles.parametersContainer}>
                        {tuning.parameter_names.map((name, idx) => (
                          <View key={idx} style={styles.parameterRow}>
                            <Text style={styles.parameterLabel}>{name}:</Text>
                            <Text style={styles.parameterValue}>
                              {typeof tuning.parameters[idx] === 'number' 
                                ? tuning.parameters[idx].toFixed(4) 
                                : tuning.parameters[idx]}
                            </Text>
                          </View>
                        ))}
                      </View>
                    )}

                    {tuning.graphHtml ? (
                      <View style={styles.graphContainer}>
                        <Text style={styles.graphTitle}>Closed-Loop Response</Text>
                        <WebView
                          originWhitelist={['*']}
                          source={{ html: tuning.graphHtml }}
                          style={styles.modelGraph}
                        />
                      </View>
                    ) : (
                      <Text style={styles.noDataText}>No tuning graph available</Text>
                    )}
                  </View>
                ))
              ) : (
                <Text style={styles.noDataText}>No tuning results available</Text>
              )}
            </View>
          </View>

          <View style={styles.navButtonsContainer}>
            <TouchableOpacity 
              style={styles.prevButton} 
              onPress={() => navigation.goBack()}
            >
              <Text style={styles.buttonText}>PREVIOUS</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={styles.downloadButton}
              onPress={handleSavePDF}
            >
              <Text style={styles.buttonText}>SAVE REPORT</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};
// Create Stack Navigator
const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="Uploader"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#3498db',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen 
          name="Uploader" 
          component={ImageUploaderScreen} 
          options={{ title: 'Step 1: Image Upload' }} 
        />
        <Stack.Screen 
          name="Step2" 
          component={Step2} 
          options={{ title: 'Step 2: Extraction Results' }} 
        />
        <Stack.Screen 
          name="Step3" 
          component={Step3} 
          options={{ title: 'Step 3: System Identification' }} 
        />
        <Stack.Screen 
          name="Step4" 
          component={Step4} 
          options={{ title: 'Step 4: Controller Tuning' }} 
        />
        <Stack.Screen 
          name="Step5" 
          component={Step5} 
          options={{ title: 'Step 5: Report' }} 
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContainer: {
    flexGrow: 1,
    padding: 20,
  },
  container: {
    flex: 1,
    alignItems: 'center',
  },
  imageButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 20,
  },
  imageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 5,
  },
  galleryButton: {
    backgroundColor: '#007AFF',
  },
  cameraButton: {
    backgroundColor: '#5856D6',
  },
  disabledButton: {
    backgroundColor: '#cccccc',
  },
  imagePreview: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    marginBottom: 20,
    resizeMode: 'contain',
    backgroundColor: '#eee',
  },
  inputCard: {
    width: '100%',
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  parametersCard: {
    width: '100%',
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
    color: '#333',
  },
  inputRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  parametersRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  inputGroup: {
    width: '48%',
  },
  parameterGroup: {
    width: '48%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  inputLabel: {
    marginBottom: 5,
    fontSize: 14,
    color: '#555',
  },
  parameterLabel: {
    fontSize: 14,
    color: '#555',
    fontWeight: 'bold',
  },
  parameterValue: {
    fontSize: 14,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 10,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
  },
  colorPickerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  colorPickerGroup: {
    width: '48%',
  },
  colorPickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 10,
    backgroundColor: '#f9f9f9',
  },
  colorPickerText: {
    fontSize: 16,
  },
  colorDropdown: {
    position: 'absolute',
    top: 50,
    left: 0,
    right: 0,
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    zIndex: 10,
    elevation: 5,
  },
  colorOption: {
    padding: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  uploadButton: {
    width: '100%',
    backgroundColor: '#34C759',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    height: '100%',
  },
  statusText: {
    marginTop: 10,
    fontSize: 14,
    color: '#333',
    textAlign: 'center',
  },
  resultSection: {
    width: '100%',
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  resultImage: {
    width: '100%',
    height: 250,
    borderRadius: 8,
    resizeMode: 'contain',
    backgroundColor: '#eee',
  },
  webview: {
    width: '100%',
    height: 200,
  },
   stepContent: {
    flex: 1,
    padding: 20,
  },
  centerContent: {
    alignItems: 'center',
  },
  card: {
    width: '100%',
    backgroundColor: 'white',
    borderRadius: 10,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    backgroundColor: '#3498db',
    padding: 15,
    borderTopLeftRadius: 10,
    borderTopRightRadius: 10,
  },
  cardTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  cardBody: {
    padding: 20,
  },
  dropdownContainer: {
    marginBottom: 20,
  },
  dropdownButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 5,
  },
  dropdownButtonText: {
    color: 'white',
    fontSize: 16,
  },
  dropdownMenu: {
    marginTop: 5,
    backgroundColor: 'white',
    borderRadius: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  dropdownItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  graphContainer: {
    height: 300,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    borderRadius: 5,
    marginTop: 20,
  },
  navButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  prevButton: {
    backgroundColor: '#95a5a6',
    padding: 15,
    borderRadius: 5,
    flex: 1,
    marginRight: 10,
  },
  nextButton: {
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 5,
    flex: 1,
    marginLeft: 10,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  // some style of step 3 and there is some at the top

  modelGraph: {
    width: '100%',
    height: 300,
    marginTop: 20,
    backgroundColor: 'transparent',
  },
  resultsContainer: {
    marginTop: 20,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  parametersContainer: {
    marginTop: 15,
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 8,
  },
  parameterRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  parameterLabel: {
    fontWeight: 'bold',
    color: '#555',
  },
  parameterValue: {
    color: '#333',
  },
  loadingContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#555',
  },
  disabledButton: {
    backgroundColor: '#cccccc',
  },
    errorContainer: {
    backgroundColor: '#ffebee',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  errorText: {
    color: '#c62828',
    textAlign: 'center',
  },
  noGraphText: {
    textAlign: 'center',
    marginVertical: 20,
    color: '#555',
  },

  // STEP 4 

  safeArea: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContainer: {
    flexGrow: 1,
    padding: 20,
  },
  stepContent: {
    flex: 1,
  },
  centerContent: {
    alignItems: 'center',
  },
  card: {
    width: '100%',
    backgroundColor: 'white',
    borderRadius: 10,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    backgroundColor: '#3498db',
    padding: 15,
    borderTopLeftRadius: 10,
    borderTopRightRadius: 10,
  },
  cardTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  cardBody: {
    padding: 20,
  },
  dropdownContainer: {
    marginBottom: 20,
  },
  dropdownButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 5,
  },
  dropdownButtonText: {
    color: 'white',
    fontSize: 16,
  },
  dropdownMenu: {
    marginTop: 5,
    backgroundColor: 'white',
    borderRadius: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  submenuContainer: {
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  criteriaItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
  },
  submenu: {
    backgroundColor: '#f9f9f9',
    paddingLeft: 20,
  },
  submenuItem: {
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  loadingContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#555',
  },
  resultsContainer: {
    marginTop: 20,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  modelGraph: {
    width: '100%',
    height: 300,
    backgroundColor: 'transparent',
  },
  noGraphText: {
    textAlign: 'center',
    marginVertical: 20,
    color: '#555',
  },
  parametersContainer: {
    marginTop: 15,
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 8,
  },
  parameterRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  parameterLabel: {
    fontWeight: 'bold',
    color: '#555',
  },
  parameterValue: {
    color: '#333',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  errorText: {
    color: '#c62828',
    textAlign: 'center',
  },
  navButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  prevButton: {
    backgroundColor: '#95a5a6',
    padding: 15,
    borderRadius: 5,
    flex: 1,
    marginRight: 10,
  },
  nextButton: {
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 5,
    flex: 1,
    marginLeft: 10,
  },
  disabledButton: {
    backgroundColor: '#cccccc',
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  // Add these new styles for the step indicator
  stepIndicatorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    paddingHorizontal: 20,
    position: 'relative',
  },
  stepCircle: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  activeStep: {
    backgroundColor: '#3498db',
  },
  completedStep: {
    backgroundColor: '#2ecc71',
  },
  stepNumber: {
    color: '#333',
    fontWeight: 'bold',
  },
  stepLine: {
    height: 2,
    width: 40,
    backgroundColor: '#ddd',
  },
  completedLine: {
    backgroundColor: '#2ecc71',
  },
  stepText: {
    position: 'absolute',
    bottom: -20,
    fontSize: 12,
    color: '#555',
    fontWeight: 'bold',
  },
  centerText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#555',
    paddingVertical: 20,
  },
  safeArea: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContainer: {
    flexGrow: 1,
    padding: 10,
  },
  container: {
    flex: 1,
    alignItems: 'center',
  },
  card: {
    width: '100%',
    backgroundColor: 'white',
    borderRadius: 10,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    backgroundColor: '#3498db',
    padding: 15,
    borderTopLeftRadius: 10,
    borderTopRightRadius: 10,
  },
  cardTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  cardBody: {
    padding: 20,
  },
  imageButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 20,
  },
  imageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 5,
  },
  galleryButton: {
    backgroundColor: '#007AFF',
  },
  cameraButton: {
    backgroundColor: '#5856D6',
  },
  disabledButton: {
    backgroundColor: '#cccccc',
    opacity: 0.7,
  },
  imagePreview: {
    width: '100%',
    height: 250,
    borderRadius: 8,
    marginBottom: 20,
    resizeMode: 'contain',
    backgroundColor: '#f9f9f9',
  },
  inputSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  inputRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  inputGroup: {
    width: '48%',
  },
  inputLabel: {
    marginBottom: 5,
    fontSize: 14,
    color: '#555',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
  },
  colorPickerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  colorPickerGroup: {
    width: '48%',
  },
  colorPickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 12,
    backgroundColor: '#f9f9f9',
  },
  colorPickerText: {
    fontSize: 16,
  },
  colorDropdown: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    zIndex: 10,
    elevation: 5,
    maxHeight: 200,
  },
  colorOption: {
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  primaryButton: {
    width: '100%',
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  statusText: {
    marginTop: 10,
    fontSize: 14,
    color: '#333',
    textAlign: 'center',
  },
  resultSection: {
    marginBottom: 25,
  },
  webview: {
    width: '100%',
    height: 250,
    backgroundColor: 'transparent',
  },
  parametersSection: {
    marginBottom: 20,
  },
  parametersGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  parameterItem: {
    width: '48%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  parameterLabel: {
    fontSize: 14,
    color: '#555',
    fontWeight: 'bold',
  },
  parameterValue: {
    fontSize: 14,
    color: '#333',
  },
  buttonContainer: {
    alignItems: 'center',
    marginTop: 20,
  },
  smallButton: {
    width: '50%',
    backgroundColor: '#3498db',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },

  // Step indicator styles
  stepIndicatorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 25,
    paddingHorizontal: 20,
    position: 'relative',
  },
  stepCircle: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  activeStep: {
    backgroundColor: '#3498db',
  },
  completedStep: {
    backgroundColor: '#2ecc71',
  },
  stepNumber: {
    color: '#333',
    fontWeight: 'bold',
  },
  activeStepNumber: {
    color: 'white',
  },
  stepLine: {
    height: 2,
    width: 40,
    backgroundColor: '#ddd',
  },
  completedLine: {
    backgroundColor: '#2ecc71',
  },
  stepText: {
    position: 'absolute',
    bottom: -20,
    fontSize: 12,
    color: '#555',
    fontWeight: 'bold',
  },

  // Loading styles
  loadingContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#555',
  },

  // Error styles
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  errorText: {
    color: '#c62828',
    textAlign: 'center',
  },
  // step 5 styles
   safeArea: {
    flex: 1,
    backgroundColor: '#f5f5f5'
  },
  scrollContainer: {
    padding: 15
  },
  container: {
    flex: 1
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 10,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  cardHeader: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee'
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold'
  },
  cardBody: {
    padding: 15
  },
  section: {
    marginBottom: 25,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee'
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10
  },
  imageRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15
  },
  imageContainer: {
    flex: 1,
    marginHorizontal: 5
  },
  imageTitle: {
    textAlign: 'center',
    marginBottom: 5,
    fontWeight: '600'
  },
  processImage: {
    width: '100%',
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#ddd'
  },
  parametersGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginTop: 10
  },
  parameterItem: {
    width: '48%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
    padding: 8,
    backgroundColor: '#f9f9f9',
    borderRadius: 5
  },
  colorParameterItem: {
    width: '48%',
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    padding: 8,
    backgroundColor: '#f9f9f9',
    borderRadius: 5
  },
  parameterLabel: {
    fontWeight: '600'
  },
  parameterValue: {
    color: '#333'
  },
  colorBox: {
    width: 20,
    height: 20,
    marginHorizontal: 5,
    borderWidth: 1,
    borderColor: '#ddd'
  },
  colorValue: {
    fontFamily: 'monospace',
    fontSize: 12
  },
  graphContainer: {
    marginTop: 15,
    height: 300
  },
  modelGraph: {
    flex: 1,
    minHeight: 300
  },
  graphTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 5
  },
  noDataText: {
    fontStyle: 'italic',
    color: '#999',
    textAlign: 'center',
    marginVertical: 15
  },
  navButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20
  },
  prevButton: {
    backgroundColor: '#6c757d',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    flex: 1,
    marginRight: 10
  },
  downloadButton: {
    backgroundColor: '#28a745',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    flex: 1
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold'
  },
  fittedParamsContainer: {
  marginTop: 15,
  marginBottom: 15,
},
parametersContainer: {
  marginVertical: 10,
},
parameterRow: {
  flexDirection: 'row',
  justifyContent: 'space-between',
  marginBottom: 5,
},
parameterLabel: {
  fontWeight: 'bold',
  color: '#555',
},
parameterValue: {
  color: '#333',
},

});