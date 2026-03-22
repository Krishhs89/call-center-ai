"""
Generate sample WAV audio files for testing the Call Center AI pipeline.

Each file is a realistic call center transcript converted to speech via gTTS,
then saved as 16-bit 16kHz mono WAV (Whisper-optimal format).

Run from repo root:
    venv/bin/python scripts/generate_sample_audio.py

Requires: gtts, pydub, ffmpeg
"""

import io
import sys
from pathlib import Path

# ── Sample call scripts ───────────────────────────────────────────────────────
# Each designed to exercise different V3 agents:
# billing.wav      → AutoTagging=billing, Compliance=Financial, CustomerProfile, Coaching
# refund.wav       → AutoTagging=refund, Escalation=high, wants_refund intent
# tech_support.wav → AutoTagging=technical_support, KB articles, SentimentAgent
# fraud.wav        → AutoTagging=fraud_security, Compliance=HIPAA/Financial, AnomalyDetection
# complaint.wav    → EscalationPrediction=critical, Sentiment=negative, Coaching priority
# account.wav      → Compliance=verification, KB=account_verification SOP

SAMPLES = [
    {
        "filename": "sample_audio_billing.wav",
        "description": "Billing dispute — double charge. Tests: Auto-tagging, Compliance (Financial), Customer Profile",
        "text": (
            "Agent: Thank you for calling customer support. My name is Sarah. How can I help you today? "
            "Customer: Hi Sarah, I'm calling because I was charged twice for my monthly subscription. "
            "I can see two identical charges of fifty nine dollars on my bank statement this month. "
            "Agent: I'm really sorry to hear that. I completely understand how concerning that must be. "
            "Let me pull up your account right now. Can you confirm your full name and the email on the account? "
            "Customer: Sure, it's James Wilson, james dot wilson at gmail dot com. "
            "Agent: Thank you James. I can see your account here. You're absolutely right — there are two charges. "
            "That appears to be a processing error on our end. I'm going to go ahead and issue an immediate refund "
            "for the duplicate charge. You should see fifty nine dollars back in your account within three to five business days. "
            "Customer: Thank you, I really appreciate that. "
            "Agent: Of course! Is there anything else I can help you with today? "
            "Customer: No that's all, thanks. "
            "Agent: My pleasure James. Have a wonderful day."
        ),
    },
    {
        "filename": "sample_audio_escalation.wav",
        "description": "Angry customer — escalation request. Tests: EscalationPrediction=critical, Sentiment=negative, Coaching",
        "text": (
            "Agent: Good afternoon, thank you for calling. How can I assist you today? "
            "Customer: I am absolutely furious right now. This is the third time I have called about the same issue "
            "and nothing has been resolved. This is completely unacceptable. "
            "Agent: I understand you're frustrated. What seems to be the issue? "
            "Customer: My internet has been down for five days. Five days! I work from home and I have lost "
            "thousands of dollars because of this. I want to speak to a supervisor immediately. "
            "Agent: Let me just check your account first before I transfer you. "
            "Customer: No. I don't want you to check anything. I want a supervisor right now. "
            "I've already told three of your agents everything. I am considering legal action if this is not resolved today. "
            "Agent: I understand sir. Let me transfer you to my supervisor right now. Please hold for just one moment. "
            "Customer: Fine. But if I get disconnected I am filing a formal complaint."
        ),
    },
    {
        "filename": "sample_audio_tech_support.wav",
        "description": "Tech support — app not working. Tests: KB articles, AutoTagging=technical, Sentiment=frustrated",
        "text": (
            "Agent: Hello, thanks for calling technical support. This call may be recorded for quality and training purposes. "
            "How can I help you today? "
            "Customer: Hi, I'm having trouble with your mobile app. It keeps crashing every time I try to make a payment. "
            "I've tried restarting my phone but it's still not working. "
            "Agent: I'm sorry to hear you're having this issue. What type of phone are you using and which version of our app do you have? "
            "Customer: I have an iPhone fourteen running iOS seventeen. I'm not sure what version the app is. "
            "Agent: No worries. Let me check for any known issues. "
            "Actually, we did have a brief technical issue this morning that's now resolved. "
            "Could you try deleting the app and reinstalling it from the App Store? "
            "Customer: OK I'll try that. Give me a moment. "
            "Alright I reinstalled it and it seems to be working now. The payment went through. "
            "Agent: That's great news! Is there anything else I can help you with today? "
            "Customer: No that fixed it, thank you so much. "
            "Agent: Perfect! Thank you for your patience. Have a great day."
        ),
    },
    {
        "filename": "sample_audio_fraud.wav",
        "description": "Fraud report — unauthorized transaction. Tests: AutoTagging=fraud_security, Compliance=Financial+verification",
        "text": (
            "Agent: Thank you for calling fraud prevention. This call is recorded. How can I help you? "
            "Customer: I just checked my account and there's an unauthorized transaction. "
            "Someone spent two hundred and forty dollars at a store I've never been to. "
            "Agent: I'm so sorry to hear that. Let me verify your identity before we proceed. "
            "Can I get your full name, date of birth, and the last four digits of your account number? "
            "Customer: Yes, it's Maria Lopez, date of birth March fifteenth nineteen eighty five, "
            "and the last four digits are seven seven four two. "
            "Agent: Thank you Maria, identity verified. I can see the transaction in question from yesterday evening. "
            "I'm going to flag this as fraudulent right now and block your current card. "
            "We'll issue you a replacement card which will arrive in five to seven business days. "
            "A full investigation will be opened and you'll receive a provisional credit within forty eight hours. "
            "Customer: Thank goodness. Thank you for acting so quickly. "
            "Agent: Absolutely, protecting your account is our top priority. "
            "You'll receive a confirmation email shortly. Is there anything else I can help with? "
            "Customer: No, that covers everything. Thank you."
        ),
    },
    {
        "filename": "sample_audio_complaint.wav",
        "description": "Billing complaint — poor service. Tests: Compliance violation (no recording consent), Anomaly, Coaching",
        "text": (
            "Agent: Hi there, how can I help you? "
            "Customer: I received my bill this month and it's two hundred dollars more than usual. "
            "Nobody told me about a price increase and I'm very unhappy about this. "
            "Agent: I can look into that for you. What's your account number? "
            "Customer: It's four four seven eight two one. "
            "Agent: OK I can see a plan upgrade was applied to your account. "
            "Customer: What? I never authorized any upgrade. Who changed my plan? "
            "I've been with your company for eight years and this is how you treat loyal customers? "
            "Agent: I'm not sure how the change was made. I can see it was done last month. "
            "Customer: This is ridiculous. I want a full refund for this month and I want my old plan restored immediately. "
            "Agent: I can restore your old plan but the refund would need to go to the billing team. "
            "Customer: I don't want to be transferred again. Can you just fix this now? "
            "Agent: I'll try my best. Let me put in the request for the refund and restore your plan. "
            "It may take a couple of days to process. "
            "Customer: Fine. But if this isn't resolved I'm cancelling my service."
        ),
    },
    {
        "filename": "sample_audio_account.wav",
        "description": "Account management — password reset. Tests: KB=account verification, Compliance=Financial, AutoTagging=account_management",
        "text": (
            "Agent: Good morning, thank you for calling account support. This call may be recorded for training purposes. "
            "My name is David, how can I help you? "
            "Customer: Hi David, I need to reset my password. I've been locked out of my account for two days. "
            "Agent: I'm happy to help with that. For security purposes I'll need to verify your identity. "
            "Can I get your full name and date of birth? "
            "Customer: Sure, it's Thomas Brown, born July eighth nineteen seventy nine. "
            "Agent: Thank you. Can you also confirm the answer to your security question: "
            "what is the name of your first pet? "
            "Customer: It's Buddy. "
            "Agent: Perfect, identity verified. I'm sending a password reset link to your registered email right now. "
            "You should receive it within the next two minutes. "
            "Customer: Great, I can already see it in my inbox. "
            "Agent: Excellent! Once you reset your password, I also recommend enabling two-factor authentication "
            "for extra security. Would you like me to walk you through that? "
            "Customer: No thanks, I can manage that myself. But thank you for the suggestion. "
            "Agent: Of course. Is there anything else I can help you with today? "
            "Customer: No, that's everything. You were very helpful, thank you. "
            "Agent: My pleasure Thomas. Have a great day!"
        ),
    },
]


def generate_wav(text: str, output_path: Path) -> None:
    """Convert text to speech (gTTS MP3) then convert to WAV via pydub/ffmpeg."""
    from gtts import gTTS
    from pydub import AudioSegment

    print(f"  Generating speech for '{output_path.name}'...")
    tts = gTTS(text=text, lang="en", slow=False)

    # Save as MP3 in memory
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    # Convert MP3 → WAV (16kHz, 16-bit mono — optimal for Whisper)
    audio = AudioSegment.from_mp3(mp3_buffer)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(str(output_path), format="wav")
    duration = len(audio) / 1000
    size_kb = output_path.stat().st_size // 1024
    print(f"  ✅ {output_path.name} — {duration:.0f}s, {size_kb} KB")


def main():
    # Determine output directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = repo_root / "data" / "sample_audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(SAMPLES)} sample WAV files → {output_dir}\n")

    for i, sample in enumerate(SAMPLES, 1):
        out_path = output_dir / sample["filename"]
        print(f"[{i}/{len(SAMPLES)}] {sample['description']}")
        try:
            generate_wav(sample["text"], out_path)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    print(f"\n✅ Done — {len(SAMPLES)} WAV files in {output_dir}")
    print("\nTo test in the app:")
    print("  1. Run: MOCK_LLM=false venv/bin/python -m streamlit run ui/streamlit_app.py")
    print("  2. In the Upload tab → Option 2 → upload any .wav file from data/sample_audio/")
    print("  3. Whisper will transcribe it, then the full V3 pipeline runs automatically")


if __name__ == "__main__":
    main()
