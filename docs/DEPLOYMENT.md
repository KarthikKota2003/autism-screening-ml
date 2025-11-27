# Deployment Guide - Autism Screening ML Application

Complete guide for deploying the Flask application to Render.com (free tier).

---

## Prerequisites

- GitHub account
- Render.com account (free, no credit card required)
- Git installed locally

---

## Step 1: Prepare Local Repository

### 1.1 Initialize Git Repository

```bash
cd d:\AutismML
git init
git add .
git commit -m "Initial commit: Autism screening ML application"
```

### 1.2 Verify Files

Ensure these files exist in root:
- `.gitignore`
- `Procfile`
- `render.yaml`
- `deliverables/UI/requirements.txt`
- `deliverables/UI/.env.example`

---

## Step 2: Create GitHub Repository

### 2.1 Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `autism-screening-ml` (or your choice)
3. Description: "ML-powered autism screening web application"
4. **Public** or **Private** (your choice)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### 2.2 Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/autism-screening-ml.git
git branch -M main
git push -u origin main
```

**Verify**: Check GitHub repository to ensure all files uploaded

---

## Step 3: Deploy to Render

### 3.1 Sign Up for Render

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub account (recommended)
4. Authorize Render to access your repositories

### 3.2 Create New Web Service

1. Click "New +" button
2. Select "Web Service"
3. Connect your GitHub repository:
   - If not connected, click "Connect account"
   - Select `autism-screening-ml` repository
4. Click "Connect"

### 3.3 Configure Web Service

**Basic Settings**:
- **Name**: `autism-screening-app` (or your choice)
- **Region**: Choose closest to you (e.g., Oregon, Frankfurt)
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Runtime**: `Python 3`

**Build & Deploy**:
- **Build Command**: `pip install -r deliverables/UI/requirements.txt`
- **Start Command**: `gunicorn app:app --chdir deliverables/UI`

**Plan**:
- Select **Free** tier

### 3.4 Set Environment Variables

Click "Advanced" → "Add Environment Variable":

1. **SECRET_KEY**:
   - Click "Generate Value" (Render will auto-generate)
   - Or manually set: `python -c "import secrets; print(secrets.token_hex(32))"`

2. **FLASK_ENV**:
   - Key: `FLASK_ENV`
   - Value: `production`

3. **PYTHON_VERSION** (optional):
   - Key: `PYTHON_VERSION`
   - Value: `3.12.0`

### 3.5 Deploy

1. Click "Create Web Service"
2. Render will:
   - Clone your repository
   - Install dependencies
   - Start the application
3. Wait 5-10 minutes for initial deployment

---

## Step 4: Verify Deployment

### 4.1 Check Build Logs

- Monitor the "Logs" tab for build progress
- Look for:
  ```
  ==> Installing dependencies...
  ==> Build successful!
  ==> Starting service...
  ```

### 4.2 Access Application

- Once deployed, Render provides a URL:
  - Format: `https://autism-screening-app.onrender.com`
- Click the URL or copy to browser
- **Note**: First load may take 30 seconds (cold start)

### 4.3 Test Functionality

1. **Home Page**: Verify 4 category cards display
2. **Age Classifier**: Test FAB button and modal
3. **Screening Form**: Fill out for each category
4. **Prediction**: Verify results page displays correctly

---

## Step 5: Troubleshooting

### Issue 1: Build Fails

**Symptom**: "Build failed" in logs

**Solutions**:
1. Check `requirements.txt` for typos
2. Verify Python version compatibility
3. Check build logs for specific error
4. Common fix: Update `scikit-learn` version

### Issue 2: Application Won't Start

**Symptom**: "Service unavailable" or 503 error

**Solutions**:
1. Check start command: `gunicorn app:app --chdir deliverables/UI`
2. Verify `app.py` exists in `deliverables/UI/`
3. Check logs for Python errors
4. Ensure `SECRET_KEY` environment variable is set

### Issue 3: Model Files Not Found

**Symptom**: "FileNotFoundError: Model file not found"

**Solutions**:
1. Verify `.pkl` files are in `deliverables/` directory
2. Check `.gitignore` doesn't exclude `.pkl` files
3. Ensure files were pushed to GitHub
4. Check `MODEL_DIR` path in `app.py`

### Issue 4: CSRF Token Missing

**Symptom**: "CSRF token missing" error

**Solutions**:
1. Ensure `flask-wtf` in `requirements.txt`
2. Verify `CSRFProtect(app)` in `app.py`
3. Check forms have `{{ csrf_token() }}` (if using WTForms)

### Issue 5: Slow Response Time

**Symptom**: Application takes 30+ seconds to respond

**Explanation**:
- Free tier spins down after 15 minutes
- First request after inactivity triggers cold start
- This is **normal** for free tier

**Solutions**:
- Upgrade to paid tier ($7/month) for always-on
- Use external monitoring service to ping app every 14 minutes
- Accept cold starts for free demo

---

## Step 6: Continuous Deployment

### Automatic Deployment

Render automatically deploys when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update: description of changes"
git push origin main

# Render automatically:
# 1. Detects push
# 2. Pulls latest code
# 3. Rebuilds application
# 4. Deploys new version
```

### Manual Deployment

From Render dashboard:
1. Go to your web service
2. Click "Manual Deploy"
3. Select branch
4. Click "Deploy"

---

## Step 7: Custom Domain (Optional)

### Free Subdomain

Render provides: `https://your-app-name.onrender.com`

### Custom Domain (Paid Plan)

1. Purchase domain (e.g., from Namecheap, GoDaddy)
2. In Render dashboard:
   - Go to "Settings" → "Custom Domains"
   - Click "Add Custom Domain"
   - Enter your domain
3. Update DNS records at your registrar:
   - Add CNAME record pointing to Render
4. Wait for DNS propagation (up to 48 hours)

---

## Step 8: Monitoring & Maintenance

### Health Checks

Render automatically monitors your application:
- HTTP health checks every 30 seconds
- Auto-restart if application crashes

### Logs

Access logs from Render dashboard:
- Real-time logs in "Logs" tab
- Filter by severity (info, warning, error)
- Download logs for analysis

### Metrics

Free tier includes basic metrics:
- Request count
- Response time
- Error rate
- Memory usage

---

## Alternative: Deploy to Railway.app

If Render doesn't work, try Railway:

### Quick Deploy

1. Go to https://railway.app
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Choose `autism-screening-ml`
5. Railway auto-detects Python
6. Set environment variables:
   - `SECRET_KEY`
   - `FLASK_ENV=production`
7. Click "Deploy"

**Note**: Railway gives $5 monthly credit (free)

---

## Alternative: Deploy to PythonAnywhere

For always-on free hosting (manual deployment):

### Setup

1. Sign up at https://www.pythonanywhere.com
2. Create free account
3. Go to "Web" tab
4. Click "Add a new web app"
5. Choose "Flask"
6. Upload files via "Files" tab or use git
7. Configure WSGI file
8. Reload web app

**Note**: No automatic GitHub deployment

---

## Production Checklist

Before going live:

- [ ] Environment variables set (SECRET_KEY, FLASK_ENV)
- [ ] Debug mode disabled (`FLASK_ENV=production`)
- [ ] CSRF protection enabled
- [ ] All tests passing
- [ ] Model files uploaded
- [ ] Custom domain configured (if applicable)
- [ ] Error pages created (404, 500)
- [ ] Monitoring enabled
- [ ] Backup strategy in place

---

## Security Best Practices

1. **Never commit `.env` file** (use `.env.example` instead)
2. **Rotate SECRET_KEY** periodically
3. **Use HTTPS only** (Render provides automatically)
4. **Monitor logs** for suspicious activity
5. **Keep dependencies updated** (`pip list --outdated`)
6. **Rate limiting** (consider adding Flask-Limiter)

---

## Cost Breakdown

### Free Tier (Render)

- **Cost**: $0/month
- **Limitations**:
  - 750 hours/month
  - Spins down after 15 minutes
  - 512MB RAM
  - Shared CPU
- **Best For**: Demo, portfolio, low-traffic

### Paid Tier (Render)

- **Cost**: $7/month
- **Benefits**:
  - Always-on (no spin-down)
  - 512MB RAM
  - Dedicated CPU
  - Custom domains
- **Best For**: Production, medium-traffic

---

## Support

**Render Documentation**: https://render.com/docs  
**Flask Documentation**: https://flask.palletsprojects.com/  
**Gunicorn Documentation**: https://docs.gunicorn.org/

---

## Next Steps

1. ✅ Deploy to Render
2. ✅ Test all functionality
3. ✅ Share URL with stakeholders
4. Consider adding:
   - User authentication
   - Data persistence (database)
   - Email notifications
   - Analytics tracking
   - A/B testing

---

**Deployment Date**: 2025-11-27  
**Platform**: Render.com (Free Tier)  
**Status**: Production-Ready
