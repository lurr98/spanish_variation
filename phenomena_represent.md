# General thoughts

In order to represent the variety of Spanish adequately, one important factors is to focus on the properties that really distinguish the different varieties from one another, i.e. it does not make sense to focus on the properties that are shared among many different regions but more on those that are unique to certain dialects (or wouldn't this be too specific?)

Major problem that could arise for distinction using the *voseo*: Spanish is a pro-drop language

Many of the peculiarities occur in vernacluar language, this could pose a problem

Group El Salvador, Guatemala & Honduras?

Southern Cone as a broad linguistic area?

Group the Antilles?

Evaluation has to take into account similarity of different dialects, e.g. Venezuela and the Antilles are sometimes more similar 


# Possible groupings of countries for simplified classification

- Antilles (Cuba, Puerto Rico, Dominican Republic)
- Middle Central America (El Salvador, Guatemala, Honduras)
- Andean region (Colombia, Ecuador, Peru, Chile (only Northern), Bolivia, Argentina (only Northwest))
- River Plate (Argentina, Uruguay, Paraguay (only parts))

# Possibly good phenomena

## *voseo* (keep in mind that *vos* can be used with multiple types of verbal morphology)

**not used in:**

- Cuba
- Dominican Republic (?)
- Mexico (except some parts of Chiapas)
- Peru (some regions)
- Puerto Rico
- Spain

**used in:**

- Argentina
- Bolivia
- Chile
- Colombia (*usted* as familiar pronoun)
- Costa Rica (*usted* as familiar pronoun)
- Ecuador
- El Salvador
- Guatemala
- Honduras
- Nicaragua
- Panama (but on the decline & *usted* is more frequent)
- Paraguay
- Peru (some regions)
- Uruguay
- Venezuela

### How to model?

simple word count (*vos* vs. *tú*)? However, Spanish is a pro-drop language, so this may be misleading. Look for the according endings additionally? Since there are different types of verbal morphology, this could also be an indicator for specific regions.


## clitic doubling:

- Argentina
- Bolivia
- Colombia
- Ecuador (with *le* as direct object clitic)
- Mexico
- Paraguay (with *le* and *les* as direct object clitics)
- Peru (but not when direct object has been fronted)

### How to model?

make use of POS tags, `po` should be the appropriate tag

## double possessives

- Bolivia
- Colombia (Pacific coast)
- Honduras (occasionally)
- Mexico (some regions)

### How to model?

make use of POS tags, `dp-` should be the appropriate tag

## use of overt subjects (some varieties use more pronouns than others)

### How to model?

make use of POS tags, simply let the number of subject pronouns be a predictor, `ps` should be the appropriate tag

## subjects before infinitives

- Argentina
- Colombia
- Cuba
- Dominican Republic
- Ecuador
- Panama
- Puerto Rico
- Venezuela

### How to model?

look out for the sequence `ps` + `vr` and use the count as predictor

## non-inverted questions

- Cuba
- Dominican Republic
- Panama (Panama City)
- Puerto Rico
- Venezuela

### How to model?

look out for the sequence `WH-pronoun` + `pronoun` + `verb` and use the count as predictor (find out appropriate POS tags)

## "intensive" *ser*

- Colombia
- Ecuador
- Panama
- Venezuela

### How to model?

look out for `verb` + `ser`, (find out appropriate POS tags)

## `indefinite / indirect article + possessive`

- El Salvador
- Guatemala
- Honduras
- Mexico
- Paraguay

### How to model?

look out for the sequence `indefinite / indirect article` + `possessive`, (find out appropriate POS tags or simply use lemma)

## frequency of specific tenses (some varieties prefer specific tenses, e.g. Peruvian Spanish uses perfective instead of preterite a lot)

### How to model?

tenses seem to be annotated to a certain extend but I'm not sure which means which, an alternative approach could be to simply make use of the verbal morphology accompanied with the POS tag `v` to verify that the token is actually a verb and count the tenses this way. Perfective ofc would have to be counted using the sequence `v` + `vps-ms` and gerund with `form of estar` + `vps-ms` (*estar* can have different tags, depending on the person, e.g. `vip-2s` or `vip-3p`)

## null direct object (correlated with clitic doubling of direct objects)

- Argentina
- Bolivia (?)
- Colombia (?)
- Ecuador
- Mexico (??????)
- Paraguay
- Peru

### How to model?

that's a tough one

## diminutives

- *-ico*
    - Colombia
    - Costa Rica
    - Cuba
- *-ito*
    - Mexico
- *-illo*
    - Mexico (Chiapas)
- *-ingo*
    - Bolivia

### How to model?

look out for the diminutive endings when the corresponding POS tag is actually a noun


## use of present subjunctive even if it is preceeded by a verb in past tense

- Argentina
- Ecuador
- Peru

### How to model?

…


## use of *vosotros* to distinguish Spain Spanish


# Possibly unique phenomena

## Argentinia

- in vernacular speech of many regions, *yo* replaces *a mí* in dative-verb constructions
- postposed *eso* to include other members of a group
- use of *grande* instead of *mucho*
- use of *ir en* to express motion

## Colombia

- in the Quechua-influenced highland region, diminiutive endings may be attached to clitic pronouns, especially in imperative constructions
- in the southern Andean region, verbal paraphrases with gerunds appear, e.g. combinations of venir + gerund but also others

## Costa Rica

- in vernacular speech, collective nouns are often build by adding the suffix *-ada*
- use of the suffix *-era* to refer to the act of doing something, items related to a general activity or a general collective

## Cuba

- *más* precedes negative words

## Dominican Republic

- *ello* is used instead of the null expletive in existential and extraposed sentences, also common is *ello sí* / *no* as an answer to questions

## Ecuador

- in vernacular speech, the diminutive suffix *-ito* is sometimes combined with demonstratives, numerals, interrogatives, gerunds, *no más* and more
- *estar* is often postposed, especially by Quechua-bilinguals
- throughout highland Ecuador, *dar* + gerund is often used as an imperative (possibly a calque from Quechua)

## Panama

- in vernacular speech, the addition of the suffix *vé* to familiar imperatives is quite common

## Peru

- combination of *muy* and *-ísimo*
- use of `object + verb` word order (from Quechua) ?

