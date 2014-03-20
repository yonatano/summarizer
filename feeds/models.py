from django.db import models

#feeds created by users OR Gist
class Feed(models.Model):
	name = models.TextField()
	tags = models.TextField()

class UserFeed(models.Model):
	name = models.TextField()
	tags = models.TextField()
	creator_id = models.TextField()
	creator_name = models.TextField()



   
 