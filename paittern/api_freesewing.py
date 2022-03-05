""" Module for transform output from mensurations.py in inputs
    for FREESEWING scripts"""

from optparse import Option
from bs4 import BeautifulSoup
from requests import get



patterns = ["Aaron", "Albert", "Bee", "Bella", "Benjamin","Bent", "Breanna", "Brian", "Bruce", "Carlita",
                 "Carlton", "Cathrin", "Charlie", "Cornelius", "Diana", "Florence", "Florent", "Holmes", "Hortensia", 
                 "Huey", "Hugo", "Jaeger", "Lunetius", "Paco", "Penelope", "Sandy", "Shin", "Simon", "Simone",
                 "Sven", "Tamiko", "Teagan", "Theo", "Tiberius", "Titan", "Trayvon", "Ursula",
                 "Wahid", "Walburga", "Waralee", "Yuri",]


url = 'https://freesewing.org/docs/patterns'


def patterns_measures():
    """ Function which return dictionnary with pattern as keys 
    and required measures as values"""
    patterns_measures = {}
    for pat in patterns: 
        patterns_measures[pat]=[]
        response = get(f'{url}/{pat.lower()}/')
        soup = BeautifulSoup(response.content, "html.parser")
        try:
            measures_soup = soup.find('h2',text='Required measurements').find_next(class_="links").find_all('a')
            for idx,measure in enumerate(measures_soup):
                patterns_measures[pat].append(measure.string)
        except (AttributeError,ValueError):
            pass
    return patterns_measures


def get_required_values(pattern):
    """ Function which return required measures in a list"""
    return patterns_measures[str(pattern)]
    


def patterns_options_fit():
    """ Function witch return dictionnary with pattern as keys 
    and a list of fit options as values"""
    patterns_options_fit = {}
    for pattern in patterns: 
        patterns_options_fit[pattern]=[]
        response = get(f'{url}/{pattern.lower()}/')
        soup = BeautifulSoup(response.content, "html.parser")
        try:
            measures_soup = soup.find('b',text='Fit').find_next(class_="links")
            for idx,measure in enumerate(measures_soup):
                 patterns_options_fit[pattern].append(measure.string)
        except (AttributeError,ValueError):
            pass
    return patterns_options_fit
    
    
def patterns_options_style():
    """ Function witch return dictionnary with pattern as keys 
    and a list of style options as values"""
    patterns_options_style = {}
    for pattern in patterns: 
        patterns_options_style[pattern]=[]
        response = get(f'{url}/{pattern.lower()}/')
        soup = BeautifulSoup(response.content, "html.parser")
        try:
            measures_soup = soup.find('b',text='Style').find_next(class_="links")
            for idx,measure in enumerate(measures_soup):
                patterns_options_style[pattern].append(measure.string)
        except (AttributeError,ValueError):
            pass
    return patterns_options_style


def patterns_options_advanced():
    """ Function witch return dictionnary with pattern as keys 
    and a list of advanced options as values"""
    patterns_options_advanced = {}
    for pattern in patterns: 
        patterns_options_advanced[pattern]=[]
        response = get(f'{url}/{pattern.lower()}/')
        soup = BeautifulSoup(response.content, "html.parser")
        try:
            measures_soup = soup.find('b',text='Advanced').find_next(class_="links")
            for idx,measure in enumerate(measures_soup):
                patterns_options_advanced[pattern].append(measure.string)
        except (AttributeError,ValueError):
            pass
    return patterns_options_advanced

