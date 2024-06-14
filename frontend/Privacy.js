import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { useFonts } from 'expo-font';
import { useNavigation } from '@react-navigation/native';



export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Topic': require('./assets/Fonts/Inter/static/Inter-Bold.ttf'),
    'Inter-Body': require('./assets/Fonts/Inter/static/Inter-Regular.ttf')
  });

  const [isChecked, setIsChecked] = useState(false);

  const navigation = useNavigation();

  if (!fontsLoaded) {
    return null;
  }

  

  const GuidelinePress = () => {
    navigation.navigate('Guidelines')
  };

  return (
    <View style={styles.container}>
      <Text style={styles.topic}>Privacy Policy:</Text>

      <Text style={styles.paragraph}>We respect your privacy and are committed to protecting your personal information. Any data collected from you will be used solely for the purpose of providing and improving our services. We assure you that your information will not be shared with third parties without your explicit consent. We implement industry-standard security measures to safeguard your data against unauthorized access and ensure its confidentiality. By using our services, you agree to the collection and use of information in accordance with this policy. Please review our Privacy Policy periodically for any updates. If you have any concerns or questions, feel free to contact us. We respect your privacy and are committed to protecting your personal information. Any data collected from you will be used solely for the purpose of providing and improving our services. We assure you that your information will not be shared with third parties without your explicit consent. We implement industry-standard security measures to safeguard your data against unauthorized access and ensure its confidentiality.</Text>

      <TouchableOpacity style={[styles.checkbox, isChecked && styles.checked]} onPress={() => setIsChecked(!isChecked)}>
        {isChecked && <Text style={styles.checkmark}>✓</Text>}
      </TouchableOpacity>

      <Text style={styles.agreeText}>I agree</Text>

      <TouchableOpacity style={styles.button} onPress={GuidelinePress}>
        <Text style={styles.buttonText}>GUIDELINES</Text>
      </TouchableOpacity>

      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#3A0BA3',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  topic: {
    fontFamily: 'Inter-Topic',
    color: '#fff',
    fontSize: 20,
    marginBottom: 35,
    textAlign: 'center',
    left: -95
  },
  paragraph: {
    fontFamily: 'Inter-Body',
    color: '#fff',
    textAlign: 'justify',
    marginBottom: 60,
    padding: 20
  },
  button: {
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 50,
    marginBottom: 20,
    bottom: 30
  },
  buttonText: {
    fontFamily: 'Inter-Topic',
    color: '#3A0BA3',
    fontSize: 20,
  },
  checkbox: {
    borderWidth: 1,
    borderColor: '#fff',
    borderRadius: 5,
    width: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: -15,
    bottom: 40,
    left: 80
  },
  checked: {
    backgroundColor: '#fff',
  },
  checkmark: {
    color: '#3A0BA3',
    fontSize: 12,
  },
  agreeText: {
    justifyContent: 'space-around',
    fontFamily: 'Inter-Body',
    color: '#fff',
    fontSize: 16,
    marginBottom: 30,
    marginLeft: 270,
    bottom: 60
  },
});