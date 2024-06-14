import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TouchableOpacity, Image } from 'react-native';
import { useFonts } from 'expo-font';
import * as ImagePicker from 'expo-image-picker';

export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Topic': require('./assets/Fonts/Inter/static/Inter-Bold.ttf'),
    'Inter-Body': require('./assets/Fonts/Inter/static/Inter-Regular.ttf')
  });

  const [isChecked, setIsChecked] = useState(false);

  if (!fontsLoaded) {
    return null;
  }

  const AllowPress = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (permissionResult.granted === false) {
        alert('Permission to access camera roll is required!');
        return;
      }
      const pickerResult = await ImagePicker.launchImageLibraryAsync();
      console.log(pickerResult);
      // Here you can do something with the picked image, such as display it in your app
    } catch (error) {
      console.error('Error accessing image picker:', error);
    }
  };
  

  return (
    <View style={styles.container}>

      <Image style={styles.cameraImage} source={require('./assets/Images/gallery 1.png')} />
      <Text style={styles.topic}>ACCESS REQUIRED</Text>

      <Text style={styles.paragraph}>The Gallery Access feature in the Humming Bird Sign Language Translator App provides users with a visually engaging repository of sign language gestures. Seamlessly integrated into the app's user interface, the Gallery offers users a comprehensive collection of clear and accurate sign representations. Users can explore and review a diverse array of sign language gestures, enhancing their understanding and proficiency. The intuitive navigation ensures easy access to specific signs, while the interactive design allows users to engage with the content at their own pace. Whether users are seeking to reinforce their learning or simply appreciate the beauty of sign language expressions, the Gallery Access feature in the Humming Bird app provides a valuable and immersive resource.</Text>
      <TouchableOpacity style={styles.button} onPress={(AllowPress)}>
        <Text style={styles.buttonText}>ALLOW</Text>
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
    marginBottom: 60,
    padding: 20
  },
  button: {
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 50,
    marginBottom: 180,
  },
  buttonText: {
    fontFamily: 'Inter-Topic',
    color: '#3A0BA3',
    fontSize: 20,
  },
  cameraImage: {
    resizeMode: 'contain',
    width: 70,
    bottom: -90
  }
  
});