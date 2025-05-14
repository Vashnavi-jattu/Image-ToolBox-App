import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TextInput,
  StyleSheet, 
  TouchableOpacity, 
  Dimensions, 
  Platform, 
  StatusBar,
  Image,
  Alert,
  ActivityIndicator,
  ScrollView
} from 'react-native';
import { registerRootComponent } from 'expo';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Picker } from '@react-native-picker/picker';
import Icon from 'react-native-vector-icons/FontAwesome';

const API_URL = 'http://127.0.0.1:8000';

type StepNavigationProps = {
  activeStep: number;
  onStepPress: (stepId: number) => void;
};

type StepProps = {
  onNext?: (results: any) => void;
  onPrevious?: () => void;
  image?: string | null;
  extractedData?: any;
  loading?: boolean;
  pickImage?: (type: 'gallery' | 'camera') => void;
  inputs?: {
    x_min: string;
    x_max: string;
    y_min: string;
    y_max: string;
    opColor: string;
    pvColor: string;
    spColor: string;
  };
  handleInputChange?: (name: string, value: string) => void;
};




const App: React.FC = () => {
  const [inputs, setInputs] = useState({
  x_min: '0',
  x_max: '1',
  y_min: '0',
  y_max: '1',
  opColor: 'red',
  pvColor: 'blue'  // Removed spColor
});

  const handleInputChange = (name: string, value: string) => {
    setInputs(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const [activeStep, setActiveStep] = useState<number>(1);
  const [image, setImage] = useState<string | null>(null);
  const [extractedData, setExtractedData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async (type: 'gallery' | 'camera') => {
    let result;
    
    if (type === 'gallery') {
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
      setImage(result.assets[0].uri);
      setExtractedData(null);
    }
  };
  const handleNextStep = async () => {
    setActiveStep(activeStep + 1);
    
  };

  const handlePreviousStep = () => {
    setActiveStep(activeStep - 1);
  };

  return (
    <View style={styles.mainContainer}>
      <StepNavigation activeStep={activeStep} onStepPress={setActiveStep} />
      
      <View style={styles.contentContainer}>
        {activeStep === 1 && (
          <Step1 
            onNext={handleNextStep}
            image={image}
            pickImage={pickImage}
            loading={loading}
            inputs={inputs}
            handleInputChange={handleInputChange}
          />
        )}
        {activeStep === 2 && (
          <Step2 
            onNext={handleNextStep}
            onPrevious={handlePreviousStep}
            extractedData={extractedData}
            image={image}
          />
        )}
        {activeStep === 3 && (
          <Step3 
            onNext={handleNextStep}
            onPrevious={handlePreviousStep}
          />
        )}
        {activeStep === 4 && (
          <Step4 
            onNext={handleNextStep}
            onPrevious={handlePreviousStep}
          />
        )}
        {activeStep === 5 && (
          <Step5 onPrevious={handlePreviousStep} />
        )}
      </View>
    </View>
  );
};

const StepNavigation: React.FC<StepNavigationProps> = ({ activeStep, onStepPress }) => {
  const steps = [
    { id: 1, title: 'Step 1:\nLoad Image' },
    { id: 2, title: 'Step 2:\nExtract Data' },
    { id: 3, title: 'Step 3:\nModeling' },
    { id: 4, title: 'Step 4:\nTuning' },
    { id: 5, title: 'Step 5:\nReport' },
  ];

  const screenWidth = Dimensions.get('window').width;
  const stepWidth = (screenWidth - 40) / steps.length;

  return (
    <View style={styles.navContainer}>
      <View style={styles.wizardSteps}>
        {steps.map((step, index) => (
          <View key={step.id} style={[styles.stepContainer, { width: stepWidth }]}>
            <Text style={[
              styles.stepTitle,
              activeStep === step.id && styles.activeStepTitle
            ]}>
              {step.title}
            </Text>
            
            <View style={styles.stepCircleWrapper}>
              {index > 0 && (
                <View style={[
                  styles.connectorLine,
                  activeStep >= step.id && styles.activeConnectorLine
                ]} />
              )}
              
              <TouchableOpacity 
                style={[
                  styles.stepCircle,
                  activeStep === step.id && styles.activeStepCircle
                ]}
                onPress={() => onStepPress(step.id)}
              >
                <Text style={styles.stepNumber}>{step.id}</Text>
              </TouchableOpacity>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
};

const Step1: React.FC<StepProps> = ({ 
  onNext, 
  image, 
  pickImage, 
  loading,
  inputs,
  handleInputChange 
}) => {
  const [showDropdown, setShowDropdown] = useState<{
    op: boolean;
    pv: boolean;
  }>({ op: false, pv: false });
  const [apiLoading, setApiLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleDropdown = (type: 'op' | 'pv') => {
    setShowDropdown(prev => ({
      op: type === 'op' ? !prev.op : false,
      pv: type === 'pv' ? !prev.pv : false,
    }));
  };

  const handleColorSelect = (type: 'opColor' | 'pvColor', value: string) => {
    handleInputChange?.(type, value);
    setShowDropdown({ op: false, pv: false });
  };

  const handleNext = async () => {
    if (!image || !inputs) return;
    
    setApiLoading(true);
    setError(null);
    
    try {
      console.log('Starting image processing...');
      
      // Convert image URI to base64
      const base64Image = await new Promise<string>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.onload = function() {
          const reader = new FileReader();
          reader.onloadend = function() {
            resolve(reader.result as string);
          };
          reader.onerror = reject;
          reader.readAsDataURL(xhr.response);
        };
        xhr.onerror = reject;
        xhr.responseType = 'blob';
        xhr.open('GET', image, true);
        xhr.send(null);
      });

      // Prepare request payload
      const payload = {
        image: base64Image,
        y_min: inputs.y_min,
        y_max: inputs.y_max,
        x_min: inputs.x_min,
        x_max: inputs.x_max,
        opColor: inputs.opColor,
        pvColor: inputs.pvColor
      };

      console.log('Sending request to Django API...');
      
      // Make API call with axios
      const response = await axios.post('http://127.0.0.1.125:8000/api/extract_data/', payload, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      });

      console.log('API response received:', response.data);

      if (response.data.status !== 'success') {
        throw new Error(response.data.message || 'API request failed');
      }

      // Pass results to parent component
      onNext?.(response.data.results);
      
    } catch (err) {
      let errorMessage = 'An unknown error occurred';
      
      if (axios.isAxiosError(err)) {
        console.error('Axios error:', err.response?.data || err.message);
        errorMessage = err.response?.data?.message || err.message;
      } else if (err instanceof Error) {
        console.error('Processing error:', err.message);
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setApiLoading(false);
    }
  };

  const colorOptions = [
    { label: 'Red', value: 'red' },
    { label: 'Blue', value: 'blue' },
    { label: 'Green', value: 'green' }
  ];

  return (
    <ScrollView contentContainerStyle={styles.stepScrollContainer}>
      <View style={styles.stepContent}>
        <View style={styles.centerContent}>
          <View style={styles.imageButtonsContainer}>
            <View style={styles.buttonGroup}>
              <TouchableOpacity 
                style={[styles.imageButton, (loading || apiLoading) && styles.disabledButton]}
                onPress={() => pickImage?.('camera')}
                disabled={loading || apiLoading}
              >
                <Text style={styles.buttonText}>Take Picture</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.imageButton, (loading || apiLoading) && styles.disabledButton]}
                onPress={() => pickImage?.('gallery')}
                disabled={loading || apiLoading}
              >
                <Text style={styles.buttonText}>Choose from Gallery</Text>
              </TouchableOpacity>
            </View>
          </View>

          {image ? (
            <>
              <Image source={{ uri: image }} style={styles.smallImagePreview} />
              
              <View style={styles.compactCard}>
                <View style={styles.cardHeader}>
                  <Text style={styles.cardTitle}>Setup Input</Text>
                </View>
                
                <View style={styles.cardBody}>
                  <View style={styles.compactInputRow}>
                    <View style={styles.compactInputWrapper}>
                      <Text style={styles.smallLabel}>X min</Text>
                      <TextInput
                        style={styles.smallInput}
                        value={inputs?.x_min}
                        onChangeText={(text) => handleInputChange?.('x_min', text)}
                        placeholder="0"
                        keyboardType="numeric"
                      />
                    </View>
                    
                    <View style={styles.compactInputWrapper}>
                      <Text style={styles.smallLabel}>X max</Text>
                      <TextInput
                        style={styles.smallInput}
                        value={inputs?.x_max}
                        onChangeText={(text) => handleInputChange?.('x_max', text)}
                        placeholder="1"
                        keyboardType="numeric"
                      />
                    </View>
                  </View>

                  <View style={styles.compactInputRow}>
                    <View style={styles.compactInputWrapper}>
                      <Text style={styles.smallLabel}>Y min</Text>
                      <TextInput
                        style={styles.smallInput}
                        value={inputs?.y_min}
                        onChangeText={(text) => handleInputChange?.('y_min', text)}
                        placeholder="0"
                        keyboardType="numeric"
                      />
                    </View>
                    
                    <View style={styles.compactInputWrapper}>
                      <Text style={styles.smallLabel}>Y max</Text>
                      <TextInput
                        style={styles.smallInput}
                        value={inputs?.y_max}
                        onChangeText={(text) => handleInputChange?.('y_max', text)}
                        placeholder="1"
                        keyboardType="numeric"
                      />
                    </View>
                  </View>

                  <View style={styles.compactColorPickersContainer}>
                    <View style={styles.colorDropdownWrapper}>
                      <Text style={styles.smallLabel}>OP Color:</Text>
                      <View style={styles.dropdownContainer}>
                        <TouchableOpacity 
                          style={styles.colorDropdownButton}
                          onPress={() => toggleDropdown('op')}
                        >
                          <Text style={styles.colorDropdownText}>{inputs?.opColor}</Text>
                          <Icon 
                            name={showDropdown.op ? "chevron-up" : "chevron-down"} 
                            size={16} 
                            color="#333" 
                          />
                        </TouchableOpacity>

                        {showDropdown.op && (
                          <View style={styles.colorDropdownMenu}>
                            {colorOptions.map(option => (
                              <TouchableOpacity
                                key={`op_${option.value}`}
                                style={styles.colorDropdownItem}
                                onPress={() => handleColorSelect('opColor', option.value)}
                              >
                                <Text>{option.label}</Text>
                              </TouchableOpacity>
                            ))}
                          </View>
                        )}
                      </View>
                    </View>

                    <View style={styles.colorDropdownWrapper}>
                      <Text style={styles.smallLabel}>PV Color:</Text>
                      <View style={styles.dropdownContainer}>
                        <TouchableOpacity 
                          style={styles.colorDropdownButton}
                          onPress={() => toggleDropdown('pv')}
                        >
                          <Text style={styles.colorDropdownText}>{inputs?.pvColor}</Text>
                          <Icon 
                            name={showDropdown.pv ? "chevron-up" : "chevron-down"} 
                            size={16} 
                            color="#333" 
                          />
                        </TouchableOpacity>

                        {showDropdown.pv && (
                          <View style={styles.colorDropdownMenu}>
                            {colorOptions.map(option => (
                              <TouchableOpacity
                                key={`pv_${option.value}`}
                                style={styles.colorDropdownItem}
                                onPress={() => handleColorSelect('pvColor', option.value)}
                              >
                                <Text>{option.label}</Text>
                              </TouchableOpacity>
                            ))}
                          </View>
                        )}
                      </View>
                    </View>
                  </View>
                </View>
              </View>
            </>
          ) : (
            <Text style={styles.centerText}>Please select an image</Text>
          )}
        </View>
        
        {(loading || apiLoading) ? (
          <ActivityIndicator size="small" style={styles.loadingIndicator} />
        ) : (
          image && (
            <View style={styles.nextButtonContainer}>
              {error && (
                <Text style={styles.errorText}>
                  Error: {error}
                </Text>
              )}
              <TouchableOpacity 
                style={[styles.nextButton, (!image || loading || apiLoading) && styles.disabledButton]} 
                onPress={handleNext}
                disabled={!image || loading || apiLoading}
              >
                <Text style={styles.buttonText}>NEXT</Text>
              </TouchableOpacity>
            </View>
          )
        )}
      </View>
    </ScrollView>
  );
};

const Step4: React.FC<StepProps> = ({ onNext, onPrevious }) => {
  const [showCriteriaDropdown, setShowCriteriaDropdown] = useState(false);
  const [selectedCriteria, setSelectedCriteria] = useState('Select Criteria');
  const [selectedController, setSelectedController] = useState('');
  const [activeSubmenu, setActiveSubmenu] = useState<string | null>(null);

  const criteriaOptions = [
    { 
      name: 'ISE', 
      controllers: ['PID', 'PI', 'P'] 
    },
    { 
      name: 'IAE', 
      controllers: ['PID', 'PI', 'P'] 
    },
    { 
      name: 'ITAE', 
      controllers: ['PID', 'PI', 'P'] 
    }
  ];

  const handleCriteriaSelect = (criteria: string) => {
    setSelectedCriteria(criteria);
    setActiveSubmenu(activeSubmenu === criteria ? null : criteria);
  };

  const handleControllerSelect = (criteria: string, controller: string) => {
    setSelectedController(`${criteria}_${controller}`);
    setShowCriteriaDropdown(false);
    setActiveSubmenu(null);
    // Add your controller tuning logic here
  };

  return (
    <View style={styles.stepContent}>
      <View style={styles.centerContent}>
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Step 4: Select Criteria</Text>
          </View>
          <View style={styles.cardBody}>
            <View style={styles.dropdownContainer}>
              <TouchableOpacity 
                style={styles.dropdownButton}
                onPress={() => setShowCriteriaDropdown(!showCriteriaDropdown)}
              >
                <Text style={styles.dropdownButtonText}>
                  {selectedController ? selectedController : selectedCriteria}
                </Text>
                <Icon name={showCriteriaDropdown ? "chevron-up" : "chevron-down"} size={20} color="#fff" />
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

            {/* Placeholder for graph container */}
            <View style={styles.graphContainer}>
              <Text>Tuning graph will appear here</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.navButtonsContainer}>
        <TouchableOpacity style={styles.prevButton} onPress={onPrevious}>
          <Text style={styles.buttonText}>PREVIOUS</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.nextButton} onPress={onNext}>
          <Text style={styles.buttonText}>NEXT</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const Step5: React.FC<StepProps> = ({ onPrevious }) => {
  return (
    <View style={styles.stepContent}>
      <View style={styles.centerContent}>
        <Text style={styles.centerText}>Step 5 Content</Text>
      </View>
      <View style={styles.navButtonsContainer}>
        <TouchableOpacity style={styles.prevButton} onPress={onPrevious}>
          <Text style={styles.buttonText}>PREVIOUS</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  navContainer: {
    paddingHorizontal: 20,
    paddingTop: Platform.select({
      ios: 60,
      android: StatusBar.currentHeight || 24,
      default: 0
    }),
    paddingBottom: 20,
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    zIndex: 1,
  },
  contentContainer: {
    flex: 1,
    paddingBottom: Platform.select({
      ios: 34,
      android: 0,
      default: 0
    }),
  },
  sectionHeader: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 10,
    marginBottom: 8,
    alignSelf: 'flex-start',
  },
  dataContainer: {
    width: '100%',
    maxHeight: 150,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 5,
    padding: 8,
    marginBottom: 10,
  },
  // step2 styles
  stepContent: {
    flex: 1,
    padding: 15,
  },
  stepScrollContainer: {
    flexGrow: 1,
    paddingBottom: 15,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  navButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  nextButton: {
    backgroundColor: '#007bff',
    padding: 12,
    borderRadius: 5,
    alignItems: 'center',
    minWidth: 120,
  },
  prevButton: {
    backgroundColor: '#6c757d',
    padding: 12,
    borderRadius: 5,
    alignItems: 'center',
    minWidth: 120,
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  wizardSteps: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    position: 'relative',
  },
  stepContainer: {
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepTitle: {
    fontSize: 12,
    marginBottom: 10,
    color: 'gray',
    textAlign: 'center',
  },
  activeStepTitle: {
    color: '#007bff',
    fontWeight: '600',
  },
  stepCircleWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  stepCircle: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 2,
  },
  activeStepCircle: {
    backgroundColor: '#007bff',
  },
  stepNumber: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  connectorLine: {
    position: 'absolute',
    left: -((Dimensions.get('window').width - 40) / (2 * 5) + 20),
    width: ((Dimensions.get('window').width - 40) / 5 - 40),
    height: 2,
    backgroundColor: '#ddd',
    zIndex: 1,
    top: '50%',
  },
  activeConnectorLine: {
    backgroundColor: 'green',
  },
  smallImagePreview: {
    width: '100%',
    height: 150,
    resizeMode: 'contain',
    marginBottom: 10,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  imageButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginTop: 10,
    marginBottom: 15,
  },
  imageButton: {
    backgroundColor: '#007bff',
    padding: 10,
    borderRadius: 5,
    minWidth: 140,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  dataText: {
    fontFamily: 'monospace',
    color: '#444',
    fontSize: 12,
  },
  disabledButton: {
    backgroundColor: '#cccccc',
  },
  loadingIndicator: {
    marginVertical: 10,
  },
  compactCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
    width: '100%',
    marginBottom: 10,
    padding: 10,
    height: "50%"
  },
  cardHeader: {
    padding: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    alignItems: 'center',
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  cardBody: {
    padding: 12,
  },
  compactInputRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  compactInputWrapper: {
    width: '48%',
  },
  smallLabel: {
    marginBottom: 4,
    fontSize: 12,
    color: '#555',
    fontWeight: '500',
  },
  smallInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 5,
    padding: 8,
    fontSize: 13,
    backgroundColor: '#f9f9f9',
    height: 38,
    
  },
  compactColorPickersContainer: {
  flexDirection: 'row',
  justifyContent: 'space-between',
  marginTop: 8,
},
colorDropdownWrapper: {
  width: '30%',
},
dropdownContainer: {
  position: 'relative',
},
colorDropdownButton: {
  flexDirection: 'row',
  alignItems: 'center',
  justifyContent: 'space-between',
  borderWidth: 1,
  borderColor: '#ddd',
  borderRadius: 4,
  padding: 8,
  backgroundColor: '#fff',
},
colorDropdownText: {
  fontSize: 12,
},
colorDropdownMenu: {
  position: 'absolute',
  top: 35,
  left: 0,
  right: 0,
  backgroundColor: '#fff',
  borderWidth: 1,
  borderColor: '#ddd',
  borderRadius: 4,
  zIndex: 10,
  elevation: 3,
},
colorDropdownItem: {
  padding: 8,
  borderBottomWidth: 1,
  borderBottomColor: '#eee',
},
  // step 3 styles
  card: {
  backgroundColor: '#fff',
  borderRadius: 8,
  shadowColor: '#000',
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.1,
  shadowRadius: 4,
  elevation: 3,
  width: '100%',
  marginBottom: 20,
},
cardHeader: {
  padding: 15,
  borderBottomWidth: 1,
  borderBottomColor: '#eee',
},
cardTitle: {
  fontSize: 18,
  fontWeight: 'bold',
  color: '#333',
},
cardBody: {
  padding: 15,
},
dropdownContainer: {
  marginBottom: 20,
  position: 'relative',
},
dropdownButton: {
  backgroundColor: '#007bff',
  padding: 12,
  borderRadius: 5,
  flexDirection: 'row',
  justifyContent: 'space-between',
  alignItems: 'center',
},
dropdownButtonText: {
  color: '#fff',
  fontSize: 16,
},
dropdownMenu: {
  position: 'absolute',
  top: 50,
  left: 0,
  right: 0,
  backgroundColor: '#fff',
  borderRadius: 5,
  shadowColor: '#000',
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.2,
  shadowRadius: 4,
  elevation: 5,
  zIndex: 10,
},
dropdownItem: {
  padding: 12,
  borderBottomWidth: 1,
  borderBottomColor: '#eee',
},
graphContainer: {
  height: 300,
  borderWidth: 1,
  borderColor: '#ddd',
  borderRadius: 5,
  justifyContent: 'center',
  alignItems: 'center',
  backgroundColor: '#f9f9f9',
},
// step 4 styles

submenuContainer: {
  width: '100%',
},
criteriaItem: {
  padding: 12,
  flexDirection: 'row',
  justifyContent: 'space-between',
  alignItems: 'center',
  borderBottomWidth: 1,
  borderBottomColor: '#eee',
},
submenu: {
  backgroundColor: '#f8f9fa',
  paddingLeft: 20,
},
submenuItem: {
  padding: 10,
  borderBottomWidth: 1,
  borderBottomColor: '#eee',
},
});

registerRootComponent(App);
export default App;