# 🌐 Step-by-Step Free Cloud Deployment Guide for Students
## Host Your Live Web Portal for Free on Render.com (0$ / No Credit Card Required)

This guide will help you launch **HybridCredit-LLM** as a live, publicly accessible website on the internet (e.g. `https://hybridcredit-llm.onrender.com`). You can open this link on your smartphone or laptop during your meeting with Dr. Siba Panda!

---

## Step 1: Push Your Project to GitHub
1. Make sure all your local changes are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Build multi-agent Dockerized risk engine"
   git push origin main
   ```

---

## Step 2: Create a Free Render Account
1. Open your web browser and go to [Render.com](https://render.com/).
2. Click **"Sign Up"** and log in with your **GitHub account**.

---

## Step 3: Deploy as a Free Web Service
1. In the Render Dashboard, click the blue **"New +"** button at the top right and select **"Web Service"**.
2. Select **"Build and deploy from a Git repository"** and click **Next**.
3. Connect your GitHub repository (`institutional-risk-engine` or `Credit-Risk-LLM-Mistral-7B`).
4. Fill in the following settings:
   - **Name:** `hybridcredit-llm` (or any custom name you prefer)
   - **Region:** Singapore or Frankfurt (or nearest region)
   - **Branch:** `main`
   - **Language / Environment:** Select **Docker** (Render will automatically detect your `Dockerfile`!).
   - **Instance Type:** Select **Free ($0/month)**.

---

## Step 4: Click "Create Web Service"
1. Scroll to the bottom and click **"Create Web Service"**.
2. Render will automatically pull your `Dockerfile`, build the Python 3.10 environment, install dependencies, and launch your Flask web portal.
3. In 2–3 minutes, Render will display a green **"Live"** status badge along with your public URL:
   `https://hybridcredit-llm.onrender.com`

---

## 🎉 You Are Live!
- Click the URL to view your live multi-page web application.
- Share this link on your smartphone before walking into Dr. Siba Panda's office. You can now demo real-time underwriting, macro stress testing, and multi-agent risk committee debates live from anywhere!
