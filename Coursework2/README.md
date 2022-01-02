# COMP0090-CW2
Repository for the Group Coursework component of COMP0090, UCL, 2021/2022

   
## Link to dataset:
https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/tree/oxpet/
## The folder system matches what we have in our repository, 
## so just put the .h5 files in the respective git folders 
## on your local system

Commands:
1. git clone https://github.com/ChrisWilkin/COMP0090-CW2.git - clones this repository to the working directory
2. git add <filename> - adds the files specified to the commit group
3. git commit -m "<description>" - commits the specified fiels to the local repository with the description message inputted
4. git commit - will promt for a multiline message. To escape, press ESC COLON(:) W Q ENTER (i think)
5. git push - uploads local changes to GitHub
6. git pull - downloads the latest version from GitHub
7. git branch <branch name> - creates new branch, seperate from the main timeline, that you can make changes to without affecting the main branch
8. git log - view list of commits in the current branch
9. git checkout <hash> - enter new branch starting from the commit specified (the hash can be retrieved from the git log). git checkout master returns to main branch
10. git merge <branch> - merges the specified branch with the main branch. PLEASE CHECK WITH EVERYONE ELSE IN THE TEAM BEFORE DOING THIS!!!
11. git - provides a list of all commands and their functions
12. git status - gives a summary of the current index (what files have been changed /waiting to commit etc.)

Weight Naming Convention:
<Network Name><Param1><Param1 Value><Param2><Param2 Value>...<><lr><learning rate value(after 0.)>ep<epoch numbers><version number>.pt
eg. for Unet with k=12, lr=0.001, 10 epochs and third version:
        Unetk12lr001ep10v3.pt
    for YOLO with lr=0.001, 20 epochs and first version:
        YOLOlr001ep20v1.pt
  


If you have any questions about this let me know on WhatsApp!!
