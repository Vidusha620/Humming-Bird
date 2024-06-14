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

  const StartPress = () => {
    navigation.navigate('UploadOrTake')
  };

  return (
    <View style={styles.container}>
      <Text style={styles.topic}>Guidelines:</Text>

      <Text style={styles.paragraph}>The Humming Bird Sign Language Translator App is designed to provide users with an intuitive and enriching learning experience. The interface prioritizes user-friendly navigation, ensuring that users can effortlessly explore the app's features. Clear and accurate representations of sign language gestures facilitate easy comprehension and learning. Interactive modules offer users the opportunity to practice and enhance their sign language skills, with customization options allowing personalization to individual preferences. The app's commitment to offline accessibility, regular content updates, and robust privacy measures underscores its dedication to inclusivity and user satisfaction. By fostering community engagement and respecting cultural nuances, the Humming Bird app aims to create a supportive environment for users on their sign language learning journey. Translator App is designed to provide users with an intuitive and enriching learning experience. The interface prioritizes user-friendly navigation, ensuring that users can effortlessly explore the app's features. Clear and accurate representations of sign language gestures facilitate easy comprehension and learning. Interactive modules offer users the opportunity to practice and enhance </Text>

      

      

      <TouchableOpacity style={styles.button} onPress={(StartPress)}>
        <Text style={styles.buttonText}>START</Text>
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
    marginBottom: 30,
    textAlign: 'center',
    left: -110
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
  },
  buttonText: {
    fontFamily: 'Inter-Topic',
    color: '#3A0BA3',
    fontSize: 20,
  },
  
});