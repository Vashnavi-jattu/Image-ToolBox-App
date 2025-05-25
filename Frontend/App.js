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