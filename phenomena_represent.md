# Possibly good phenomena

## *voseo* (keep in mind that *vos* can be used with multiple types of verbal morphology)

**not used in:**

- Cuba
- Dominican Republic (?)
- Mexico (except some parts of Chiapas)
- Peru (some regions)
- Puerto Rico

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

simple word count (*vos* vs. *tú*)?


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