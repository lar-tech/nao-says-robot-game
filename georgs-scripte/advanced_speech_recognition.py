#!/usr/bin/env python
"""
Enhanced NAO Speech Recognition Script

This script demonstrates advanced speech recognition features:
- Multiple vocabulary words
- Real-time result monitoring with confidence levels
- Audio feedback from the robot
- Proper error handling and status messages
- Interactive control with keyboard interrupt
"""

import sys
import time
import signal

# Following tutorial: import ALProxy directly for speech recognition
from naoqi import ALProxy

# Add inao to path for NAO class (for ALMemory access and TTS)
sys.path.insert(0, "/home/georg/Desktop/hands_on_nao/inao")
from inao import NAO


class EnhancedSpeechRecognitionDemo:
    def __init__(self, nao_ip="192.168.1.118"):
        self.nao_ip = nao_ip

        # Create ALSpeechRecognition proxy (following tutorial exactly)
        self.asr = ALProxy("ALSpeechRecognition", nao_ip, 9559)

        # Create NAO instance for ALMemory access and TTS feedback
        self.nao = NAO(nao_ip)
        self.tts = self.nao.tts

        # Enhanced vocabulary with more useful commands
        self.vocabulary = [
            "hello",
            "goodbye",
            "thank you",
            "please",
            "yes",
            "no",
            "maybe",
            "stop",
            "forward",
            "backward",
            "left",
            "right",
            "turn around",
            "what is your name",
            "how are you",
            "nice to meet you",
            "red",
            "blue",
            "green",
            "yellow",
            "one",
            "two",
            "three",
            "four",
            "five",
        ]

        # Control flags
        self.running = False
        self.subscriber_name = "Enhanced_ASR"
        self.last_recognition_time = 0

    def setup_recognition(self):
        """Set up speech recognition with enhanced features"""
        print("=== Setting up Enhanced Speech Recognition ===")

        try:
            # Check available languages
            available_languages = self.asr.getAvailableLanguages()
            print("Available languages:", available_languages)

            # Set language
            self.asr.setLanguage("English")
            current_lang = self.asr.getLanguage()
            print("Language set to:", current_lang)

            # Set vocabulary with word spotting disabled
            self.asr.setVocabulary(self.vocabulary, False)
            print("Vocabulary loaded with", len(self.vocabulary), "words:")
            for i, word in enumerate(self.vocabulary):
                print("  {:2d}. {}".format(i + 1, word))

            # Enable audio feedback (bip sounds)
            self.asr.setAudioExpression(True)
            print("Audio feedback enabled (bip sounds)")

            # Enable visual feedback (LED animations)
            self.asr.setVisualExpression(True)
            print("Visual feedback enabled (LED animations)")

            # Set sensitivity parameter (optional enhancement)
            self.asr.setParameter("Sensitivity", 0.7)  # Balanced sensitivity
            print("Sensitivity set to 0.7")

            print("✓ Setup complete!")
            return True

        except Exception as e:
            print("✗ Setup failed:", str(e))
            return False

    def start_recognition(self):
        """Start the speech recognition engine"""
        print("\n=== Starting Speech Recognition ===")

        try:
            # Subscribe to start recognition
            self.asr.subscribe(self.subscriber_name)
            self.running = True
            print("✓ Speech recognition engine started")

            # Audio welcome message
            self.tts.say(
                "Enhanced speech recognition started. Try saying hello, goodbye, or give me a command."
            )

            return True

        except Exception as e:
            print("✗ Failed to start recognition:", str(e))
            return False

    def stop_recognition(self):
        """Stop the speech recognition engine"""
        print("\n=== Stopping Speech Recognition ===")

        try:
            self.asr.unsubscribe(self.subscriber_name)
            self.running = False
            print("✓ Speech recognition engine stopped")

            # Goodbye message
            self.tts.say("Speech recognition stopped. Goodbye.")

        except Exception as e:
            print("✗ Error stopping recognition:", str(e))

    def process_recognized_words(self):
        """Process and display recognized words with enhanced analysis"""
        try:
            # Get recognition results from ALMemory
            word_data = self.nao.memory.getData("WordRecognized")

            if word_data and len(word_data) >= 2:
                current_time = time.time()

                # Avoid processing the same recognition multiple times
                if current_time - self.last_recognition_time < 1.0:
                    return False

                self.last_recognition_time = current_time

                # Parse recognition results
                recognized_words = []
                for i in range(0, len(word_data), 2):
                    if i + 1 < len(word_data):
                        word = str(word_data[i])
                        confidence = float(word_data[i + 1])
                        recognized_words.append((word, confidence))

                if recognized_words:
                    # Sort by confidence (best first)
                    recognized_words.sort(key=lambda x: x[1], reverse=True)
                    best_word, best_confidence = recognized_words[0]

                    print("\n" + "=" * 50)
                    print("🎤 SPEECH RECOGNIZED!")
                    print("=" * 50)
                    print(
                        "Best match: '{}' (confidence: {:.1f}%)".format(
                            best_word, best_confidence * 100
                        )
                    )

                    # Confidence level interpretation
                    if best_confidence > 0.8:
                        confidence_level = "Very High"
                        confidence_color = "🟢"
                    elif best_confidence > 0.6:
                        confidence_level = "High"
                        confidence_color = "🟡"
                    elif best_confidence > 0.4:
                        confidence_level = "Medium"
                        confidence_color = "🟠"
                    else:
                        confidence_level = "Low"
                        confidence_color = "🔴"

                    print(
                        "Confidence level: {} {}".format(
                            confidence_color, confidence_level
                        )
                    )

                    # Show alternative hypotheses
                    if len(recognized_words) > 1:
                        print("\nAlternative matches:")
                        for i, (word, conf) in enumerate(recognized_words[1:4], 1):
                            print("  {}. '{}' ({:.1f}%)".format(i, word, conf * 100))

                    # Intelligent response based on recognized word
                    self.respond_to_command(best_word, best_confidence)

                    return True

        except Exception as e:
            print("Error processing recognition:", str(e))

        return False

    def respond_to_command(self, word, confidence):
        """Generate intelligent responses based on recognized commands"""
        word_lower = word.lower()

        # High confidence responses
        if confidence > 0.6:
            if word_lower in ["hello", "hi"]:
                self.tts.say("Hello! Nice to meet you!")
                print("🤖 Response: Greeting returned")

            elif word_lower in ["goodbye", "bye"]:
                self.tts.say("Goodbye! Have a great day!")
                print("🤖 Response: Farewell given")

            elif word_lower == "thank you":
                self.tts.say("You're welcome!")
                print("🤖 Response: Politeness acknowledged")

            elif word_lower == "please":
                self.tts.say("How can I help you?")
                print("🤖 Response: Offering assistance")

            elif word_lower in ["yes", "no"]:
                self.tts.say("I understand you said " + word_lower)
                print("🤖 Response: Confirmation acknowledged")

            elif word_lower == "stop":
                self.tts.say("Stopping as requested")
                print("🤖 Response: Stop command acknowledged")
                self.running = False

            elif word_lower in ["forward", "backward", "left", "right"]:
                self.tts.say("I would move " + word_lower + " if I had motion control")
                print("🤖 Response: Movement command noted (simulation)")

            elif word_lower == "what is your name":
                self.tts.say("My name is NAO")
                print("🤖 Response: Name given")

            elif word_lower == "how are you":
                self.tts.say("I'm doing well, thank you for asking!")
                print("🤖 Response: Status shared")

            elif word_lower in ["red", "blue", "green", "yellow"]:
                self.tts.say("I see you like the color " + word_lower)
                print("🤖 Response: Color preference noted")

            elif word_lower in ["one", "two", "three", "four", "five"]:
                number_words = {
                    "one": "1",
                    "two": "2",
                    "three": "3",
                    "four": "4",
                    "five": "5",
                }
                self.tts.say("You said the number " + number_words[word_lower])
                print("🤖 Response: Number recognized")

            else:
                self.tts.say("I heard you say " + word)
                print("🤖 Response: Generic acknowledgment")

        else:
            # Low confidence - ask for clarification
            self.tts.say(
                "I'm not sure I heard that correctly. Could you please repeat?"
            )
            print("🤖 Response: Requesting clarification (low confidence)")

    def run(self):
        """Main recognition loop with enhanced features"""

        def signal_handler(sig, frame):
            print("\n🛑 Interrupt received - stopping recognition...")
            self.stop_recognition()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        print("🎙️  Enhanced NAO Speech Recognition Demo")
        print("=" * 50)
        print("This demo includes:")
        print("• Real-time speech recognition")
        print("• Confidence level analysis")
        print("• Intelligent robot responses")
        print("• Multiple vocabulary words")
        print("• Audio and visual feedback")
        print("=" * 50)

        # Setup phase
        if not self.setup_recognition():
            print("❌ Setup failed - exiting")
            return

        # Start recognition
        if not self.start_recognition():
            print("❌ Failed to start recognition - exiting")
            return

        # Main recognition loop
        print("\n🎧 Listening... Try saying any of these words:")
        print(", ".join(self.vocabulary))
        print("\n💡 Tips:")
        print("• Speak clearly and at normal volume")
        print("• Stand 1-2 meters from the robot")
        print("• Say 'stop' to end the demo")
        print("• Press Ctrl+C to quit immediately")
        print("\n" + "=" * 50)

        try:
            while self.running:
                # Check for new recognitions
                self.process_recognized_words()

                # Small delay to prevent busy waiting
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_recognition()


def main():
    print("🚀 Starting Enhanced NAO Speech Recognition Demo")

    # Configuration
    ROBOT_IP = "192.168.1.118"

    # Create and run the demo
    demo = EnhancedSpeechRecognitionDemo(ROBOT_IP)
    demo.run()


if __name__ == "__main__":
    main()
