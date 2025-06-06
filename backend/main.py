from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio  # Import asyncio for adding delay

app = FastAPI(title="ASHA AI Chatbot")

# Add CORS middleware to handle preflight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat")
async def chat(request: ChatRequest):
    # Add a 2-3 second delay to simulate LLM processing
    await asyncio.sleep(2.5)  # 2.5 seconds delay

    message = request.message.strip()

    # Hardcoded input-output pairs
    if message == "find ml jobs in pune":
        # List of job strings
        job_strings = [
            "AI-ML Engineer at Left Right Mind | Pune, IN | Check details | Apply",
            "Pune || AI ML - Technical Lead IRC260159 at Hitachi Careers | Pune, IN | Check details | Apply",
            "Intern- Machine Learning (Gen AI) at Seagate Technology | Pune, IN | Check details | Apply",
            "Data Scientist -AI ML at Virtusa | Pune, IN | Check details | Apply",
            "AI Developer (Fresher) JBU at Hitachi Careers | Pune, IN | Check details | Apply",
            "AI/ML Lead at Emerson | Pune, IN | Check details | Apply",
            "Senior GenAI Engineer at micro1 | Pune, IN | Check details | Apply",
            "Python AI/ML Developer Job in Pune, India at Virtusa | Pune, IN | Check details | Apply",
            "Machine Learning Internship in Pune at Vizuara | Pune, IN | Check details | Apply",
            "Lead AI Engineer - Machine Learning at ZS | Pune, IN | Check details | Apply"
        ]

        # Parse each job string into a structured object
        jobs = []
        for job in job_strings:
            # Split the string by " | " to separate the parts
            parts = job.split(" | ")
            # The first part is "<job_title> at <employer_name>"
            title_and_employer = parts[0].split(" at ")
            job_title = title_and_employer[0].strip()
            employer_name = title_and_employer[1].strip() if len(title_and_employer) > 1 else "Unknown"
            # The second part is "<city>, <state>"
            location = parts[1].split(", ")
            job_city = location[0].strip()
            job_state = location[1].strip() if len(location) > 1 else "Unknown"
            # We don’t have apply links for the demo, so set to null
            job_apply_link = None

            # Create the job object
            jobs.append({
                "job_title": job_title,
                "employer_name": employer_name,
                "job_city": job_city,
                "job_state": job_state,
                "job_apply_link": job_apply_link
            })

        return {"jobs": jobs}

    elif message == "how to to restart your career as a woman after marriage":
        output = """Restarting your career after marriage — especially as a woman — can feel overwhelming, but it's very doable with the right steps.
Here’s a clear path you can follow:

1. Reflect on What You Want Now
Interests change over time. Ask yourself:

"What do I enjoy now?"
"What lifestyle do I want?"
"Do I want a flexible job, a remote role, or am I ready for full-time office work?"

Think about industries you genuinely like (tech, marketing, education, healthcare, fashion, etc.).

2. Assess Your Skills
List what skills you already have (communication, writing, coding, management, etc.).

Identify what skills are missing for the roles you want. (If needed, plan to upskill with online courses, certifications, or short workshops.)

3. Build a Fresh Resume and LinkedIn
Highlight transferable skills (teamwork, leadership, problem-solving, etc.).

You can mention a "career break" confidently and explain it simply:

"Career break to focus on family responsibilities, now ready to return with renewed energy and focus."

4. Start Small to Build Confidence
Freelancing, internships, part-time jobs, consulting — these are great starting points.

If you’re worried about gaps or rusty skills, this builds real recent experience without overwhelming pressure.

5. Network, Network, Network
Reach out to old colleagues, college friends, family friends.
Just let them know you're looking — you’d be surprised how often opportunities come through personal networks.

Attend local events, online webinars, workshops in your field.

6. Consider Women-Focused Platforms
Some companies and platforms specialize in helping women restart careers (e.g.,

JobsForHer (India),

HerSecondInnings (India),

Path Forward (global),

Women Back to Work).

7. Practice Interviewing
Prepare to confidently explain your career break.
It's not a weakness — it's just part of your journey.

8. Stay Kind to Yourself
Imposter syndrome is very common when restarting.
Remember: You’re not starting from scratch; you're starting from experience.

Would you like me to also suggest a few career ideas depending on your background and current interests? 🌟
(Example: remote jobs, creative fields, high-paying jobs, flexible ones, etc.)"""
        return {"guidance": output}

    elif message == "what are upcoming events on herkeys website":
        output = """# Featured Events from HerKey

## Advance Your Career with a Premier Executive MBA from Great Lakes
- **Type**: Featured
- **Date**: 16th Apr, 2025 to 30th Apr, 2025 10:00am to 6:00pm
- **Description**: Career Development
- **Link**: [https://events.herkey.com/events/advance-your-career-with-a-premier-executive-mba-from-great-lakes/1669](https://events.herkey.com/events/advance-your-career-with-a-premier-executive-mba-from-great-lakes/1669)

## Back to Work - RestartHer with MS office
- **Type**: Featured
- **Date**: 16th Apr, 2025 to 14th May, 2025 12:00pm to 11:00pm
- **Description**: 
- **Link**: [https://events.herkey.com/events/back-to-work-restarther-with-ms-office/1476](https://events.herkey.com/events/back-to-work-restarther-with-ms-office/1476)

## SkillReBoot program: Restart Your Career Journey
- **Type**: Featured
- **Date**: 21st Apr, 2025 to 20th May, 2025 6:00pm to 11:00pm
- **Description**: Career Development
- **Link**: [https://events.herkey.com/events/skillreboot-program-restart-your-career-journey/1663](https://events.herkey.com/events/skillreboot-program-restart-your-career-journey/1663)"""
        return {"guidance": output}

    elif message == "search over the internet to find tech meetups in pune":
        output = """Here are some upcoming tech meetups and conferences in Pune that you might find interesting:

---

### 🔹 **May 2025**

- **Global Azure 2025 – Pune (Virtual)**
  - 📅 Date: Saturday, May 10, 2025
  - 🕓 Time: 4:30 AM UTC
  - 📍 Format: Virtual
  - 🎯 Focus: Microsoft Azure technologies
  - 🔗 More Info:  ([Pune Tech Community - Meetup](https://www.meetup.com/pune-tech-community/?utm_source=chatgpt.com)) ([Pune Tech Community - Meetup](https://www.meetup.com/pune-tech-community/?utm_source=chatgpt.com), [Technology Events near Pune, IN - Meetup](https://www.meetup.com/find/in--pune/technology/?utm_source=chatgpt.com), [Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/Pune/tech?utm_source=chatgpt.com))

- **Clean Code: The Next Level**
  - 📅 Dates: May 8–9, 2025
  - 🏷️ Type: Advanced certification training for Java developers
  - 🔗 Details:  ([Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/Pune/tech?utm_source=chatgpt.com)) ([Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/MA/Pune/tech?utm_source=chatgpt.com), [Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/Pune/tech?utm_source=chatgpt.com))

- **The Principal Dev – Masterclass for Tech Leads**
  - 📅 Dates: May 22–23, 2025
  - 🏷️ Type: Masterclass for experienced software engineers
  - 🔗 Details:  ([Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/Pune/tech?utm_source=chatgpt.com)) ([Tech conferences in Pune 2025 / 2026 - dev.events](https://dev.events/AS/IN/Pune/tech?utm_source=chatgpt.com))

---

### 🔹 **June 2025**

- **Clean Architecture Masterclass**
  - 📅 Dates: June 5–6, 2025
  - 🏷️ Type: Certification masterclass for experienced Java developers
  - 🔗 Details:  ([Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/Pune/tech?utm_source=chatgpt.com)) ([Tech meetups in Pune 2025 / 2026 - dev.events](https://dev.events/meetups/AS/IN/Pune/tech?utm_source=chatgpt.com), [Tech conferences in Pune 2025 / 2026 - dev.events](https://dev.events/AS/IN/Pune/tech?utm_source=chatgpt.com))

---

### 🔹 **Ongoing Tech Training & Workshops**

- **Python Programming Course**
  - 📅 Duration: January 3, 2024 – July 4, 2025
  - 📍 Location: Shivajinagar, Pune
  - 💰 Fee: Free
  - 🔗 Details:  ([Upcoming Tech Events in Pune Tickets Today, This Weekend & Month](https://www.townscript.com/in/pune/tech?utm_source=chatgpt.com)) ([Upcoming Tech Events in Pune Tickets Today, This Weekend & Month](https://www.townscript.com/in/pune/tech?utm_source=chatgpt.com))

- **Data Science Training**
  - 📅 Duration: February 22 – September 22, 2025
  - 📍 Location: Kharadi, Pune
  - 💰 Fee: Free
  - 🔗 Details:  ([Upcoming Tech Events in Pune Tickets Today, This Weekend & Month](https://www.townscript.com/in/pune/tech?utm_source=chatgpt.com)) ([Pune Tech Community](https://globalai.community/communities/pune-tech-community/?utm_source=chatgpt.com), [Upcoming Tech Events in Pune Tickets Today, This Weekend & Month](https://www.townscript.com/in/pune/tech?utm_source=chatgpt.com))

- **Machine Learning Training**
  - 📅 Duration: June 1, 2022 – June 8, 2030
  - 📍 Location: Pune
  - 💰 Fee: Free
  - 🔗 Details:  ([Upcoming Tech Events in Pune Tickets Today, This Weekend & Month](https://www.townscript.com/in/pune/tech?utm_source=chatgpt.com))

---

### 🔹 **Community Tech Groups in Pune**

- **Pune Tech Community**
  - 🎯 Focus: Microsoft Azure, AI, Power Platform, and Microsoft 365
  - 🔗 Join:  ([Pune Tech Community - Meetup](https://www.meetup.com/pune-tech-community/?utm_source=chatgpt.com)) ([Pune Tech Community - Meetup](https://www.meetup.com/pune-tech-community/?utm_source=chatgpt.com))

- **Pune Tech Talks**
  - 🎯 Focus: Development methodologies, modern programming principles, containerization, and cloud development
  - 🔗 Join:  ([Pune Tech Talks - Meetup](https://www.meetup.com/pune-tech-talks/?utm_source=chatgpt.com)) ([Pune Tech Talks - Meetup](https://www.meetup.com/pune-tech-talks/?utm_source=chatgpt.com))

---

For a broader list of tech events, you can explore platforms like [Meetup](https://www.meetup.com/find/in--pune/technology/), [Eventbrite](https://www.eventbrite.com/d/india--pune/tech/), and [Townscript](https://www.townscript.com/in/pune/tech). ([Technology Events near Pune, IN - Meetup](https://www.meetup.com/find/in--pune/technology/?utm_source=chatgpt.com))

Let me know if you need more details on any specific event or assistance with registrations! """
        return {"guidance": output}

    else:
        return {"guidance": "Sorry, I can only handle specific predefined queries for this demo. Try 'find ml jobs in pune' or 'how to restart your career as a woman after marriage'."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
