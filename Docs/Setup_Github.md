
## 🚀 Setup Github
Follow this once (10–15 min). After this you can collaborate and manage code easily.

### 0.1 👤 Create & Prep Your Account
1. Go to [https://github.com](https://github.com) and Sign Up / Log In.
2. Complete your profile (photo, short bio, location) – it increases trust when others view your repositories.
3. (Optional but good) Enable 2FA under Settings > Security.

### 0.2 📦 Create or Join a Repository
To collaborate, you need a repository:
1. Click the "+" icon (top-right) → New repository.
2. Fill in name, description, and choose public/private.
3. (Optional) Add a README and .gitignore.
4. Click "Create repository".
5. To join an existing repo, ask for an invite or fork it.

#### 0.2.1 ✉️ Accepting Repository Invitations

If someone invites you to collaborate on a repository, you will usually receive an email from Github:

1. **Check your email inbox** for a Github invitation. The email will contain a link to accept the invitation.
2. Alternatively, visit [https://github.com/notifications](https://github.com/notifications) to view pending invitations.
3. Click the link in the email or notification.
4. On the repository page, click **"Accept invitation"**.
5. You now have access—refresh the repo page or clone it as described above.

### 0.3 🖥️ Install & Configure Git Locally

You can work in Google Colab, VS Code, or any local editor. Below are steps for local setup:

#### macOS / Linux

```bash
sudo apt update
sudo apt install git
git --version
```

#### Windows

1. Download and install Git from [https://git-scm.com](https://git-scm.com).
2. Open Git Bash (installed with Git).
3. Verify installation:
    ```bash
    git --version
    ```

### 0.4 📝 Set Up Your Identity
Configure your name and email (used for commits):

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### 0.5 🔗 Connect to Github (HTTPS or SSH)
#### A. 🔒 HTTPS (Recommended for beginners)
1. Go to your repository page → Click "Code" → Copy HTTPS link.
2. Clone with:
    ```bash
    git clone https://github.com/username/repo.git
    ```
3. When pushing, enter your Github username and password (or personal access token if prompted).

#### B. 🗝️ SSH (Advanced, for frequent use and recommended most of the time)

#### SSH Setup by Operating System

**//TODO reference add_ssh_github docs for more information??????????**

##### 🐧 Linux / 🍏 macOS

1. **Generate a new SSH key:**
    ```bash
    ssh-keygen -t ed25519 -C "your@email.com"
    ```
    - When prompted, press Enter to accept the default file location.
    - Optionally, set a passphrase for added security.

2. **Start the ssh-agent in the background:**
    ```bash
    eval "$(ssh-agent -s)"
    ```

3. **Add your SSH private key to the ssh-agent:**
    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```

4. **Display your public key (copy this to Github):**
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```

5. **Add the public key to Github:**  
   Go to Github → Settings → SSH and GPG keys → New SSH key.

---

##### 🪟 Windows

1. **Open Git Bash** (install from [https://git-scm.com](https://git-scm.com) if needed).

2. **Generate a new SSH key:**
    ```bash
    ssh-keygen -t ed25519 -C "your@email.com"
    ```
    - Press Enter to accept the default file location.
    - Optionally, set a passphrase.

3. **Start the ssh-agent:**
    ```bash
    eval "$(ssh-agent -s)"
    ```

4. **Add your SSH private key to the ssh-agent:**
    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```

5. **Display your public key (copy this to Github):**
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```

6. **Add the public key to Github:**  
   Go to Github → Settings → SSH and GPG keys → New SSH key.

3. Clone with:
    ```bash
    git clone git@github.com:username/repo.git
    ```

### 0.6 ✅ Quick Test
Check connection and basic commands:
```bash
git status
git pull
git push
```

💡 Tip: For collaboration, always pull before you push to avoid conflicts.

### 0.7 🛠️ Basic Git Commands

Here are some essential Git commands for daily use:

```bash
# Check the status of your repo. IMPORTANT always do this before pulling or pushing
git status

# Add files to staging. IMPORTANT avoid adding unnecessary files or using `git add .` unless you know what are you doing.
git add <filename>

# Commit changes with a message. For recommended commit conventions, see: https://www.conventionalcommits.org/en/v1.0.0/
git commit -m "Your commit message"

# Pull latest changes from remote
git pull

# Push your changes to remote
git push

# View commit history
git log

# See differences between files
git diff

# Create a new branch
git branch <branchname>

# Switch to a branch
git checkout <branchname>

# Merge a branch into current branch
git merge <branchname>
```

### 0.8 💡 Tips & Tricks

- **Use .gitignore**: Create a `.gitignore` file in your repo to specify files and directories that should be ignored by Git (e.g., `node_modules`, `.env`).

- **Stash Changes**: If you need to switch branches but have uncommitted changes, use `git stash` to save your changes temporarily.

- **View Changes Before Committing**: Use `git diff` to see changes in your working directory before staging them.

- **Interactive Rebase**: Use `git rebase -i HEAD~n` to interactively edit your last n commits (e.g., to squash commits).

- **Use Branches**: Create a new branch for each feature or bug fix to keep your work organized and separate from the main codebase.

- **Commit Often**: Make small, frequent commits with clear messages to make it easier to track changes and revert if necessary.

- **Pull Before Push**: Always pull the latest changes from the remote repository before pushing your changes to avoid conflicts.

