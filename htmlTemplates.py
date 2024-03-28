css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.cleanpng.com%2Fpng-robotics-six-degrees-of-freedom-delta-robot-techno-6139864%2F&psig=AOvVaw2cLUx1QdJ2VA_gYl_aYMMQ&ust=1711539118453000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNi4o5zqkYUDFQAAAAAdAAAAABAJ">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.stockio.com%2Ffree-icon%2Fblossom-icon-set-user&psig=AOvVaw0Ci24XmyFXIfhm5zImIoZo&ust=1711539041264000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCLCjz_rpkYUDFQAAAAAdAAAAABAE">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
