# flake8: noqa: E501

import pathlib

import femr.ontology


def create_fake_athena(tmp_path: pathlib.Path) -> pathlib.Path:
    athena = tmp_path / "athena"
    athena.mkdir()

    concept = athena / "CONCEPT.csv"
    concept.write_text(
        """concept_id	concept_name	domain_id	vocabulary_id	concept_class_id	standard_concept	concept_code	valid_start_date	valid_end_date	invalid_reason
37200198\tType 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, right eye\tCondition\tICD10CM\t7-char billing code\t\tE11.3211\t19700101\t20991231\t
380097\tMacular edema due to diabetes mellitus\tCondition\tSNOMED\tClinical Finding\tS\t312912001\t20020131\t20991231
4334884	Disorder of macula due to diabetes mellitus	Condition	SNOMED	Clinical Finding	S	232020009	20020131	20991231
4174977	Retinopathy due to diabetes mellitus	Condition	SNOMED	Clinical Finding	S	4855003	20020131	20991231
4208223	Disorder of macula of retina	Condition	SNOMED	Clinical Finding	S	312999006	20020131	20991231
4290333	Macular retinal edema	Condition	SNOMED	Clinical Finding	S	37231002	20020131	20991231
35626904	Retinal edema due to diabetes mellitus	Condition	SNOMED	Clinical Finding	S	770323005	20180731	20991231
45757435	Mild nonproliferative retinopathy due to type 2 diabetes mellitus	Condition	SNOMED	Clinical Finding	S	138911000119106	20150131	20991231
"""
    )
    relationship = athena / "CONCEPT_RELATIONSHIP.csv"
    relationship.write_text(
        """concept_id_1	concept_id_2	relationship_id	valid_start_date	valid_end_date	invalid_reason
37200198	380097	Maps to	20171001	20991231
37200198	1567956	Is a	20170428	20991231
37200198	1567959	Is a	20170428	20991231
37200198	45757435	Maps to	20171001	20991231
37200198	1567961	Is a	20170428	20991231
37200198	45552385	Is a	20170428	20991231
35977781	35977781	Mapped from	20200913	20991231
46135811	40642538	Has status	20220128	20991231
46135811	35631990	Has Module	20220128	20991231"""
    )

    ancestor = athena / "CONCEPT_ANCESTOR.csv"
    ancestor.write_text(
        """ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\tmax_levels_of_separation
373499	4334884	4	6
442793	4334884	3	3
255919	4334884	6	9
433128	4334884	4	4
4180628	4334884	6	8
4209989	4334884	3	3
441840	4334884	6	10
4274025	4334884	5	9
4082284	4334884	2	2
4042836	4334884	5	7
443767	4334884	2	2
4038502	4334884	5	8
4174977	4334884	1	1
4334884	4334884	0	0
4134440	4334884	5	7
4247371	4334884	5	8
375252	4334884	3	5
4027883	4334884	4	4
4208223	4334884	1	1
378416	4334884	2	2
4080992	4334884	4	6
4162092	4334884	3	3
255919	380097	7	10
433128	380097	5	5
442793	380097	4	4
373499	380097	5	7
4180628	380097	7	9
4209989	380097	4	4
37018677	380097	3	3
441840	380097	5	11
380097	380097	0	0
4274025	380097	4	10
372903	380097	2	2
433595	380097	4	4
4042836	380097	6	8
4082284	380097	3	3
443767	380097	3	3
4038502	380097	6	9
4174977	380097	2	2
4334884	380097	1	1
4134440	380097	6	8
4247371	380097	6	9
4290333	380097	1	1
375252	380097	4	6
4027883	380097	5	5
4040388	380097	3	3
4208223	380097	2	2
35626904	380097	1	1
378416	380097	3	3
4080992	380097	5	7
4162092	380097	4	4
"""
    )

    return athena


def test_only_athena(tmp_path: pathlib.Path) -> None:
    fake_athena = create_fake_athena(tmp_path)

    ontology = femr.ontology.Ontology(str(fake_athena))

    assert (
        ontology.get_description("ICD10CM/E11.3211")
        == "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema, right eye"
    )

    assert ontology.get_parents("ICD10CM/E11.3211") == {"SNOMED/312912001", "SNOMED/138911000119106"}
    assert ontology.get_parents("SNOMED/312912001") == {"SNOMED/37231002", "SNOMED/232020009", "SNOMED/770323005"}

    assert ontology.get_all_parents("ICD10CM/E11.3211") == {
        "SNOMED/37231002",
        "SNOMED/138911000119106",
        "SNOMED/4855003",
        "SNOMED/312999006",
        "SNOMED/312912001",
        "ICD10CM/E11.3211",
        "SNOMED/770323005",
        "SNOMED/232020009",
    }

    assert ontology.get_children("SNOMED/312912001") == {"ICD10CM/E11.3211"}
    assert ontology.get_children("SNOMED/37231002") == {"SNOMED/312912001"}

    assert ontology.get_all_children("SNOMED/37231002") == {"ICD10CM/E11.3211", "SNOMED/312912001", "SNOMED/37231002"}


def test_athena_and_custom(tmp_path: pathlib.Path) -> None:
    fake_athena = create_fake_athena(tmp_path)

    code_metadata = {
        "CUSTOM/CustomDiabetes": {"description": "A nice diabetes code", "parent_codes": ["ICD10CM/E11.3211"]}
    }

    ontology = femr.ontology.Ontology(str(fake_athena), code_metadata)

    assert ontology.get_description("CUSTOM/CustomDiabetes") == "A nice diabetes code"
    assert ontology.get_all_parents("CUSTOM/CustomDiabetes") == {
        "CUSTOM/CustomDiabetes",
        "SNOMED/37231002",
        "SNOMED/138911000119106",
        "SNOMED/4855003",
        "SNOMED/312999006",
        "SNOMED/312912001",
        "ICD10CM/E11.3211",
        "SNOMED/770323005",
        "SNOMED/232020009",
    }

    assert ontology.get_all_children("SNOMED/37231002") == {
        "CUSTOM/CustomDiabetes",
        "ICD10CM/E11.3211",
        "SNOMED/312912001",
        "SNOMED/37231002",
    }
