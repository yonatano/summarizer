from django.db import models

# Create your models here.
class Summary(models.Model):
	title = models.TextField()
	summary = models.TextField()
	original_article_text = models.TextField()
	original_article_url = models.TextField()
	publication_date = models.DateField()
