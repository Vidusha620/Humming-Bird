import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TouchableOpacity, Image } from 'react-native';
import { useFonts } from 'expo-font';

export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Topic': require('./assets/Fonts/Inter/static/Inter-Bold.ttf'),
    'Inter-Body': require('./assets/Fonts/Inter/static/Inter-Regular.ttf')
  });

  const [isChecked, setIsChecked] = useState(false);

  if (!fontsLoaded) {
    return null;
  }

  return (
    <View style={styles.container}>

      <Image style={styles.cameraImage} source={require('./assets/Images/warning 1.png')} />
      <Text style={styles.topic}>PLEASE RE-ENTER</Text>

      <Text style={styles.paragraph}>We can’t detect your images and videos. Try to find more lightened area! We can’t detect your images and videos. Try to find more lightened area! We can’t detect your images and videos. Try to find more lightened area! We can’t detect your images and videos. Try to find more lightened area! </Text>
      <TouchableOpacity style={styles.button} onPress={() => console.log("Button Pressed")}>
        <Text style={styles.buttonText}>RE-TAKE</Text>
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
  },
  paragraph: {
    fontFamily: 'Inter-Body',
    color: '#fff',
    textAlign: 'justify',
    marginBottom: 30,
    padding: 20
  },
  button: {
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 50,
    marginBottom: 260,
  },
  buttonText: {
    fontFamily: 'Inter-Topic',
    color: '#3A0BA3',
    fontSize: 20,
  },
  cameraImage: {
    resizeMode: 'contain',
    width: 70,
    bottom: -100
  }
  
});