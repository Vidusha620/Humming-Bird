import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, View, Image, TouchableOpacity } from 'react-native';
import { useFonts } from 'expo-font';
import { useNavigation } from '@react-navigation/native';


import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

import PrivacyScreen from './Privacy';
import GuidelinesScreen from './Guidelines';
import UploadScreen from './UploadOrTake';
import CameraScreen from './CameraAccess';
import GalleryScreen from './GalleryAccess';

const Stack = createStackNavigator();

export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Topic': require('./assets/Fonts/Inter/static/Inter-Bold.ttf'),
    'Inter-Body': require('./assets/Fonts/Inter/static/Inter-Regular.ttf')
  });

  if (!fontsLoaded) {
    return null;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ headerShown: false }} // Hide the header for the Home screen
        />
        <Stack.Screen name="Privacy" component={PrivacyScreen} />
        <Stack.Screen name="Guidelines" component={GuidelinesScreen} />
        <Stack.Screen name="UploadOrTake" component={UploadScreen} />
        <Stack.Screen name="CameraAccess" component={CameraScreen} />
        <Stack.Screen name="GalleryAccess" component={GalleryScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

// Define the HomeScreen component here
function HomeScreen() {
  const navigation = useNavigation();

  const logoPress = () => {
    navigation.navigate('Privacy');
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.button}
        onPress={logoPress} // Call handleButtonPress function when button is pressed
      >
        <Image style={styles.logo} source={require('./assets/Images/logo.png')} />
      </TouchableOpacity>

      <StatusBar style="light" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#3A0BA3',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    resizeMode: 'contain',
    width: 200
  },
  // button: {
  //   resizeMode: 'cover',
  //   width: 100,
  //   backgroundColor: 'white',
  //   marginBottom: 100
  // }
});