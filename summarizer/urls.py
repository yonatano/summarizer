from django.conf.urls import patterns, include, url
from summarizer.views import summarize_text, summarize_url, mainPage, getWordList, getSources
from sources_scan.views import scan_sources, list_articles, list_articles_json, categorize, getSimilarWords, article_categorization, categorize, return_corpus, train_classifier, getAverageTermFrequency, categorize_tfidf, scikittfidf, categorize_tfidf, processClustersForArticle, clusterArticles
from feeds.views import getBasicFeedWithCategory, getCustomFeed
# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'summarizer.views.home', name='home'),
    # url(r'^summarizer/', include('summarizer.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
    url(r'^summarize/url/(.*)/$', summarize_url),
    url(r'^summarize/text/(.*)', summarize_text),
    url(r'^sources/(.*)', getSources),
    url(r'^$', list_articles),
    url(r'^wordlist/', getWordList),
    url(r'^scansources/$', scan_sources),
    url(r'^news/$', list_articles),
    url(r'^news/json$', list_articles_json),
    url(r'^identify/', getSimilarWords),
    url(r'^scrape_categories/', article_categorization),
    url(r'^categorize_article/(.*)', categorize_tfidf),
    url(r'^train_classifier/', train_classifier),
    url(r'^getcorpus/', return_corpus),
    url(r'^tfidf/$', scikittfidf),
    url(r'^cosim/$', processClustersForArticle),
    url(r'^cluster$', clusterArticles),
    url(r'^feeds/custom/(.*)$',  getCustomFeed),
    url(r'^feeds/(.*)$',  getBasicFeedWithCategory),
    url(r'^cat/(.*)$', categorize_tfidf),
    url(r'^scrapegoogle/$', article_categorization),
)
