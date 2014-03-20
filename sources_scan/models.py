from django.db import models

class Cluster(models.Model):
    cluster_date = models.DateTimeField(default=None, null=True, blank=True)


class Article(models.Model):
    # title text summary url favicon image pub_date
    title = models.TextField()
    text = models.TextField()
    summary = models.TextField()
    url = models.TextField()
    favicon = models.TextField()
    image = models.TextField()
    category = models.TextField()
    pub_date = models.DateTimeField()
    cluster = models.ForeignKey(Cluster, default=None, null=True, blank=True)




class Categorized_Labeled_Article(models.Model):
	label = models.TextField()
	text = models.TextField()
	url = models.TextField()