RELATION_TYPES = [
    # TACRED
    "org:alternate_names",  # [('org', 'organization'), ('org', 'alternate_name')],                CHECK
    # "org:city_of_headquarters",  # [('org', 'organization'), ('loc', 'city')],                     CHECK
    # "org:country_of_headquarters",  # [('org', 'organization'), ('loc', 'country')],               CHECK
    "org:dissolved",  # [('org', 'organization'), ('date', 'dissolved')],                          CHECK
    "org:founded",  # [('org', 'organization'), ('date', 'founded')],                              CHECK
    "org:founded_by",  # [('org', 'organization'), ('per', 'founder')],                            CHECK
    "org:member_of",  # [('org', 'member'), ('org', 'organization')],                              CHECK
    "per:member_of"
    "org:members",  # [('org', 'organization'), ('org', 'member')],                                CHECK
    "org:number_of_employees/members",  # [('org', 'organization'), ('num', 'num_employees')],     CHECK
    "org:parents",  # [('org', 'daughter_company'), ('org', 'parent_company')],                    CHECK
    "org:political/religious_affiliation",  # [('org', 'organization'), ('misc', 'affiliation')],  CHECK
    "org:shareholders",  # [('org', 'organization'), ('org', 'shareholder')],                      CHECK
    # "org:stateorprovince_of_headquarters",  # [('org', 'organization'), ('loc', 'location')],      CHECK
    "org:subsidiaries",  # [('org', 'organization'), ('org', 'subsidiary')],                       CHECK
    "org:top_members/employees",  # [('org', 'organization'), ('per', 'member')],                  CHECK
    "org:website",  # [('org', 'organization'), ('url', 'website')],                       OPEN
    "per:age",  # [('per', 'person'), ('num', 'age')],                                             CHECK
    "per:alternate_names",  # [('per', 'person'), ('per', 'alternate_name')],                      CHECK
    "per:cause_of_death",  # [('per', 'person'), ('misc', 'cause_of_death')],              OPEN
    "per:charges",  # [('per', 'person'), ('misc', 'charge')],                             OPEN
    "per:children",  # [('per', 'parent'), ('per', 'child')],                                      CHECK
    # "per:cities_of_residence",  # [('per', 'person'), ('loc', 'city')],                            CHECK
    # "per:city_of_birth",  # [('per', 'person'), ('loc', 'city')],                                  CHECK
    # "per:city_of_death",  # [('per', 'person'), ('loc', 'city')],                                  CHECK
    # "per:countries_of_residence",  # [('per', 'person'), ('loc', 'country')],                      CHECK
    # "per:country_of_birth",  # [('per', 'person'), ('loc', 'country')],                            CHECK
    # "per:country_of_death",  # [('per', 'person'), ('loc', 'country')],                            CHECK
    "per:date_of_birth",  # [('per', 'person'), ('date', 'date_of_birth')],                        CHECK
    "per:date_of_death",  # [('per', 'person'), ('date', 'date_of_death')],                        CHECK
    "per:employee_of",  # [('per', 'employee'), ('org', 'employer')],                              CHECK
    "per:origin",  # [('per', 'person'), ('loc', 'origin')],                                       CHECK
    "per:other_family",  # [('per', 'person'), ('per', 'family_member')],                          CHECK
    "per:parents",  # [('per', 'person'), ('per', 'parent')],                                      CHECK
    "per:religion",  # [('per', 'person'), ('misc', 'religion')],                                  CHECK
    "per:schools_attended",  # [('per', 'person'), ('org', 'school')],                             CHECK
    "per:siblings",  # [('per', 'person'), ('per', 'person')],                                     CHECK
    # "per:stateorprovince_of_birth",  # [('per', 'person'), ('loc', 'location')],                   CHECK
    # "per:stateorprovince_of_death",  # [('per', 'person'), ('loc', 'location')],                   CHECK
    # "per:stateorprovinces_of_residence",  # [('per', 'person'), ('loc', 'location')],              CHECK
    "per:spouse",  # [('per', 'spouse'), ('per', 'spouse')],                                       CHECK
    "per:title",  # [('per', 'person'), ('misc', 'title')],  TODO OPEN title is mostly job positions, rename??
    "no_relation",  # [('none', 'none'), ('none', 'none')]
    "org:place_of_headquarters",     # as superclass for city/country_of_headquarters
    # KnowledgeNet
    "per:political_affiliation",  # (per, org)
    "per:place_of_birth",  # as a superclass for city/stateorprovince/country
    "per:place_of_death",  # as a superclass for city/stateorprovince/country, not acutally in KNET CHECK
    "per:places_of_residence",  # as a superclass for city/stateorprovince/country
    # GIDS
    "per:degree",  # (per, degree)   list of degree names can be obtained from GIDS file!     OPEN
    # DOCRED
    "loc:capital_of",  # (loc, loc/gpe) P1376, P36                                           CHECK
    "per:conflict",  # (per, loc) P607                                                             CHECK
    "loc:located_in",  # (loc, loc) P131  , P706      ? P206                                               CHECK
    "per:language",  # (per, language) P1412                                                       CHECK
    # "publisher",  # TODO naming (org/per, work_of_art) P123                                                 CHECK
    "org:location_of_formation",  # (org, loc) P740                                                   CHECK
    "per:head_of_gov/state",  # (gpe, per)         P6, P35                                            CHECK
    # "location",  # TODO naming (fac/event/item, loc) P276                                                  CHECK
    "per:country_of_citizenship",  # (per, loc/gpe) P27                                               CHECK
    "per:notable_work",  # (per, work of art) P800                                                    CHECK
    "org:production_company",  # (org, work of art) P272                                              CHECK
    "per:creator",  # (per, work of art) P170                                                         CHECK
    "per:ethnic_group",  # (per, NORP?) P172                                                          CHECK
    # "org:manufacturer",  # (org, prod) P176, P1056  originally "manufacturer, product or material produced"   CHECK
    # -> SDW "org:product_or_technology_or_service"
    # "position held",  # (per, position) P39         -> TACRED per:title                                  OPEN
    "per:producer",  # (per, work of art) P162                                                        CHECK
    "loc:contains_location",  # (loc/gpe, loc/gpe) P150                                               CHECK
    "per:author",  # (per, work of art) P50                                                           CHECK
    "per:director",  # (per, work of art) P57                                                         CHECK
    "per:work_location",  # (per, loc) P937                                                           CHECK
    "per:religion",  # (per, norp) P140                                                               CHECK
    "loc:unemployment_rate",  # (loc/gpe, number)                                                     CHECK
    "loc:country_of_origin",  # P495 (loc, work of art/misc?)                   OPEN
    "per:performer",  # P175                                                                          CHECK
    "per:composer",  # per/work_of_art, P86                                                          CHECK
    "per:lyrics_by",  # P676                                                                          CHECK
    "per:director",  # P57                                                                            CHECK
    "per:screenwriter",  # P58                                                                        CHECK
    "per:developer",  # P178                                                              OPEN
    "loc:twinned_adm_body",  # P190  (loc, loc)   sister city                                                   CHECK
    # Fewrel
    # "parent",  # P22 actually father, P25 mother -> per:parent
    # "member of political party",  # -> TACRED per:political_affiliation
    # "hq location",  # P159 -> org:place_of_headquarters
    # "sibling",  # P3373 -> per:siblings
    "loc:country",  # P17 (of item)
    # "occupation",  # P106 (per, job)   -> TACRED position/title
    # "residence",  # P551 (per, loc)   -> per:places_of_residence
    # "subsidiary",  # P355 (parent, subsidiary) -> org:subsidiary
    # "owned by",  # P127 (org, per/org) -> TACRED shareholders)
    "loc:location_of",  # P276 (event/work_of_art/misc, loc)
    "per:field_of_work",  # P101 (per, misc) mix of profession/field of work
    # NYT
    # "/business/business_location/parent_company",
    # "/business/company/founders",
    # "/business/company/industry",
    # "/business/company/locations",
    # "/business/company/major_shareholders",
    # "/business/company/place_founded",
    # "/business/company_shareholder/major_shareholder_of",
    # "/business/person/company",
    # "/location/administrative_division/country",
    # "/location/br_state/capital",
    # "/location/cn_province/capital",
    # "/location/country/administrative_divisions",
    # "/location/country/capital",
    # "/location/de_state/capital",
    # "/location/fr_region/capital",
    # "/location/in_state/administrative_capital",
    # "/location/in_state/judicial_capital",
    # "/location/in_state/legislative_capital",
    # "/location/it_region/capital",
    # "/location/jp_prefecture/capital",
    # "/location/location/contains",
    # "/location/mx_state/capital",
    # "/location/neighborhood/neighborhood_of",
    # "/location/province/capital",
    # "/location/us_county/county_seat",
    # "/location/us_state/capital",
    # "/people/deceased_person/place_of_burial",
    # "/people/deceased_person/place_of_death",
    # "/people/family/country",
    # "/people/family/members",
    # "/people/person/children",
    # "/people/person/ethnicity",
    # "/people/person/nationality",
    # "/people/person/place_lived",
    # "/people/person/place_of_birth",    # TACRED "per:(country|city|state_or_province)_of_birth",
    # "/people/person/profession",    # TACRED position, title
    # "/people/person/religion",
    # "/people/place_of_interment/interred_here",
    # "/people/profession/people_with_this_profession",
    # "/people/ethnicity/people",
    # "/time/event/locations",
    # SDW
    "org:product_or_technology_or_service",
    # (org, product/technology/service)  Docred P176, P1056; FewRel P176, ProductCorpus of DFKI           CHECK
    "org:facility_or_location",  # (org, fac/loc) maybe Docred P706? P276, FewRel P276                     CHECK
    "org:acquisition",  # (org, org)                                                                 CHECK
    "loc:event_or_disaster",  # (loc, disaster-type/event)                            CHECK
    "org:insolvency",  # (org, cause_of_insolvency)  or rumor of insolvency?                OPEN
    "org:date_of_insolvency",  # (org, date)                                                         CHECK
    "org:layoffs",  # (org, loc)                                                                 CHECK
    "org:merger",  # (org, org)                                                                 CHECK
    "org:spinoff",  # (org, org)                                                                 CHECK
    "org:strike",  # (org, loc)                                                                 CHECK
    "org:turnover",  # (org, number/money)                                                        CHECK
    "org:revenue",  # (org, number/money)                                                        CHECK
    "org:industry",  # (org, industry)                                                    CHECK
    "org:customer",  # (org, org)                                                                 CHECK
    "org:fin_event",  # (org, fin_event)    fin-events -> spree gazetteer                  OPEN
]
