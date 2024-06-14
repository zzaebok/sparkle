import os
from invoke import task
from dotenv import load_dotenv

load_dotenv()


def make_output_dir(output_dir, lr, batch_size, num_beams, ablation_mode="full"):
    assert ablation_mode in ["full", "retrieval", "wocd"]
    if ablation_mode == "retrieval":
        output_dir = os.path.join(
            output_dir, f"ablation/lr{lr}_batch{batch_size}_beam{num_beams}"
        )
    elif ablation_mode == "wocd":
        output_dir = os.path.join(
            output_dir, f"wocd/lr{lr}_batch{batch_size}_beam{num_beams}"
        )
    else:
        output_dir = os.path.join(
            output_dir, f"lr{lr}_batch{batch_size}_beam{num_beams}"
        )
    return output_dir


@task
def preprocess_webqsp(
    ctx,
    model_name_or_path="facebook/bart-large",
    dataset_dir=f"{os.getenv('DATA_PATH')}/original/webqsp",
    kg_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/freebase",
    output_dir=f"{os.getenv('DATA_PATH')}/preprocessed/webqsp",
):
    script = f"""\\
        python3 scripts/preprocessing/dataset/preprocess_dataset.py --dataset=webqsp \\
        --model_name_or_path={model_name_or_path} \\
        --dataset_dir={dataset_dir} \\
        --kg_preprocess_dir={kg_preprocess_dir} \\
        --output_dir={output_dir} \\
        """
    ctx.run(script, echo=True)


@task
def preprocess_lcquad1(
    ctx,
    model_name_or_path="facebook/bart-large",
    dataset_dir=f"{os.getenv('DATA_PATH')}/original/lcquad1",
    kg_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/dbpedia",
    output_dir=f"{os.getenv('DATA_PATH')}/preprocessed/lcquad1",
):
    script = f"""\\
        python3 scripts/preprocessing/dataset/preprocess_dataset.py --dataset=lcquad1 \\
        --model_name_or_path={model_name_or_path} \\
        --dataset_dir={dataset_dir} \\
        --kg_preprocess_dir={kg_preprocess_dir} \\
        --output_dir={output_dir} \\
        """
    ctx.run(script, echo=True)


@task
def preprocess_simplequestions(
    ctx,
    model_name_or_path="facebook/bart-large",
    dataset_dir=f"{os.getenv('DATA_PATH')}/original/simplequestions",
    kg_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/wikidata",
    output_dir=f"{os.getenv('DATA_PATH')}/preprocessed/simplequestions",
):
    script = f"""\\
        python3 scripts/preprocessing/dataset/preprocess_dataset.py --dataset=simplequestions \\
        --model_name_or_path={model_name_or_path} \\
        --dataset_dir={dataset_dir} \\
        --kg_preprocess_dir={kg_preprocess_dir} \\
        --output_dir={output_dir} \\
        """
    ctx.run(script, echo=True)


@task
def train_sparkle(
    ctx,
    model_name_or_path,
    train_file,
    eval_file,
    test_file,
    ent_rel_token_dict,
    ent_rel_tries,
    link_dicts,
    output_dir,
    learning_rate,
    epochs,
    batch_size,
    eval_steps,
    max_eval_samples,
    num_beams,
    generation_max_length,
    var_tokens,
    select_token,
    ask_token,
):
    script = f"""\\
        mkdir -p {output_dir} && \\
        python3 scripts/training/train_sparkle.py --save_total_limit=2 \\
            --model_name_or_path={model_name_or_path} --lr_scheduler_type=linear \\
            --warmup_steps=1000 --load_best_model_at_end=True --save_steps={eval_steps} \\
            --metric_for_best_model=eval_f1 --do_train --do_eval --do_predict \\
            --greater_is_better=True --evaluation_strategy=steps \\
            --predict_with_generate=True --generation_num_beams={num_beams} --generation_max_length={generation_max_length} \\
            --train_file={train_file} \\
            --validation_file={eval_file} \\
            --test_file={test_file} \\
            --ent_rel_token_dict_path={ent_rel_token_dict} \\
            --ent_rel_tries_path={ent_rel_tries} \\
            --link_dicts_path={link_dicts} \\
            --output_dir={output_dir} --learning_rate={learning_rate} --num_train_epochs={epochs} \\
            --per_device_train_batch_size={batch_size} --per_device_eval_batch_size={batch_size} \\
            --eval_steps={eval_steps} --max_eval_samples={max_eval_samples} \\
            --select_token='{select_token}' --ask_token='{ask_token}' --var_tokens {" ".join([f"'{t}'" for t in var_tokens])} \\
            2>&1 | tee {output_dir}/train.log
        """
    ctx.run(script, echo=True)


@task
def train_lcquad1(
    ctx,
    model_name_or_path="facebook/bart-large",
    data_path=f"{os.getenv('DATA_PATH')}",
    suffix_path="",
    lr=5e-5,
    epochs=30,
    batch_size=4,
    eval_steps=100,
    max_eval_samples=200,
    num_beams=4,
    generation_max_length=128,
):
    preprocessed_dir = os.path.join(data_path, "preprocessed/lcquad1" + suffix_path)
    output_dir = make_output_dir(
        os.path.join(data_path, f"output/lcquad1/{model_name_or_path}"),
        lr,
        batch_size,
        num_beams,
    )
    train_file = os.path.join(preprocessed_dir, "train_data.json")
    eval_file = os.path.join(preprocessed_dir, "eval_data.json")
    test_file = os.path.join(preprocessed_dir, "test_data.json")
    ent_rel_token_dict = os.path.join(
        preprocessed_dir, "entity_relation_token_dict.pkl"
    )
    ent_rel_tries = os.path.join(preprocessed_dir, "entity_relation_tries.pkl")
    link_dicts = os.path.join(preprocessed_dir, "link_dictionaries.pkl")
    train_sparkle(
        ctx,
        model_name_or_path,
        train_file,
        eval_file,
        test_file,
        ent_rel_token_dict,
        ent_rel_tries,
        link_dicts,
        output_dir,
        lr,
        epochs,
        batch_size,
        eval_steps,
        max_eval_samples,
        num_beams,
        generation_max_length,
        ["var_uri ", "var_x "],
        "SELECT DISTINCT ",
        "ASK ",
    )


@task
def train_webqsp(
    ctx,
    model_name_or_path="facebook/bart-large",
    data_path=f"{os.getenv('DATA_PATH')}",
    suffix_path="",
    lr=5e-5,
    epochs=30,
    batch_size=4,
    eval_steps=100,
    max_eval_samples=200,
    num_beams=4,
    generation_max_length=128,
):
    preprocessed_dir = os.path.join(data_path, "preprocessed/webqsp" + suffix_path)
    output_dir = make_output_dir(
        os.path.join(data_path, f"output/webqsp/{model_name_or_path}"),
        lr,
        batch_size,
        num_beams,
    )
    train_file = os.path.join(preprocessed_dir, "train_data.json")
    eval_file = os.path.join(preprocessed_dir, "eval_data.json")
    test_file = os.path.join(preprocessed_dir, "test_data.json")
    ent_rel_token_dict = os.path.join(
        preprocessed_dir, "entity_relation_token_dict.pkl"
    )
    ent_rel_tries = os.path.join(preprocessed_dir, "entity_relation_tries.pkl")
    link_dicts = os.path.join(preprocessed_dir, "link_dictionaries.pkl")
    train_sparkle(
        ctx,
        model_name_or_path,
        train_file,
        eval_file,
        test_file,
        ent_rel_token_dict,
        ent_rel_tries,
        link_dicts,
        output_dir,
        lr,
        epochs,
        batch_size,
        eval_steps,
        max_eval_samples,
        num_beams,
        generation_max_length,
        [
            "var_x ",
            "var_y ",
            "var_sk0 ",
        ],
        "SELECT DISTINCT ",
        "ASK ",
    )


# set SkipValidationCallback 7000 of training simplequestions for faster training
@task
def train_simplequestions(
    ctx,
    model_name_or_path="facebook/bart-large",
    data_path=f"{os.getenv('DATA_PATH')}",
    suffix_path="",
    lr=5e-5,
    epochs=20,
    batch_size=4,
    eval_steps=100,
    max_eval_samples=200,
    num_beams=4,
    generation_max_length=128,
):
    preprocessed_dir = os.path.join(
        data_path, "preprocessed/simplequestions" + suffix_path
    )
    output_dir = make_output_dir(
        os.path.join(data_path, f"output/simplequestions/{model_name_or_path}"),
        lr,
        batch_size,
        num_beams,
    )
    train_file = os.path.join(preprocessed_dir, "train_data.json")
    eval_file = os.path.join(preprocessed_dir, "eval_data.json")
    test_file = os.path.join(preprocessed_dir, "test_data.json")
    ent_rel_token_dict = os.path.join(
        preprocessed_dir, "entity_relation_token_dict.pkl"
    )
    ent_rel_tries = os.path.join(preprocessed_dir, "entity_relation_tries.pkl")
    link_dicts = os.path.join(preprocessed_dir, "link_dictionaries.pkl")
    train_sparkle(
        ctx,
        model_name_or_path,
        train_file,
        eval_file,
        test_file,
        ent_rel_token_dict,
        ent_rel_tries,
        link_dicts,
        output_dir,
        lr,
        epochs,
        batch_size,
        eval_steps,
        max_eval_samples,
        num_beams,
        generation_max_length,
        ["var_x ", "var_uri"],
        "SELECT ",
        "ASK ",
    )


@task
def generate_sparkle(
    ctx,
    model_name_or_path,
    data_file,
    batch_size,
    num_beams,
    kg_preprocess_dir,
    var_tokens,
    select_token,
    ask_token,
    ablation_mode="full",
):
    output_path = os.path.join(
        model_name_or_path,
        f"generated/{ablation_mode + '_' if ablation_mode != 'full' else ''}generate_batch{batch_size}_beam{num_beams}.json",
    )
    script = f"""\\
        python3 scripts/evaluation/generate_sparkle.py \\
            --model_name_or_path={model_name_or_path} \\
            --data_file={data_file} \\
            --batch_size={batch_size} --num_beams={num_beams} \\
            --kg_preprocess_dir={kg_preprocess_dir} \\
            --select_token='{select_token}' --ask_token='{ask_token}' --var_tokens {" ".join([f"'{t}'" for t in var_tokens])} \\
            --ablation_mode={ablation_mode} \\
            --output_path={output_path} \\
        """
    ctx.run(script, echo=True)
    return output_path


@task
def generate_simplequestions(
    ctx,
    model_name_or_path,
    data_file=f"{os.getenv('DATA_PATH')}/preprocessed/simplequestions/test_data.json",
    batch_size=8,
    num_beams=10,
    kg_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/simplequestions",
    include_evaluate=False,
    ablation_mode="full",
):
    predict_file = generate_sparkle(
        ctx,
        model_name_or_path,
        data_file,
        batch_size,
        num_beams,
        kg_preprocess_dir,
        ["var_x ", "var_uri"],
        "SELECT ",
        "ASK ",
        ablation_mode,
    )

    if include_evaluate:
        evaluate_simplequestions(ctx, predict_file)


@task
def generate_lcquad1(
    ctx,
    model_name_or_path,
    data_file=f"{os.getenv('DATA_PATH')}/preprocessed/lcquad1/test_data.json",
    batch_size=8,
    num_beams=10,
    kg_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/lcquad1",
    include_evaluate=False,
    ablation_mode="full",
):
    predict_file = generate_sparkle(
        ctx,
        model_name_or_path,
        data_file,
        batch_size,
        num_beams,
        kg_preprocess_dir,
        ["var_uri ", "var_x "],
        "SELECT DISTINCT ",
        "ASK ",
        ablation_mode,
    )

    if include_evaluate:
        evaluate_lcquad1(ctx, predict_file)


@task
def generate_webqsp(
    ctx,
    model_name_or_path,
    data_file=f"{os.getenv('DATA_PATH')}/preprocessed/webqsp/test_data.json",
    batch_size=8,
    num_beams=10,
    kg_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/webqsp",
    include_evaluate=False,
    ablation_mode="full",
):
    predict_file = generate_sparkle(
        ctx,
        model_name_or_path,
        data_file,
        batch_size,
        num_beams,
        kg_preprocess_dir,
        [
            "var_x ",
            "var_y ",
            "var_sk0 ",
        ],
        "SELECT DISTINCT ",
        "ASK ",
        ablation_mode,
    )

    if include_evaluate:
        evaluate_webqsp(ctx, predict_file)


@task
def evaluate_sparkle(
    ctx,
    dataset,
    predict_file,
    dataset_preprocess_dir,
    endpoint,
    cold_start,
    warm_start,
):
    cold_start = "--cold_start" if cold_start else ""
    warm_start = "--warm_start" if warm_start else ""
    script = f"""\\
        python3 scripts/evaluation/evaluate_sparkle.py \\
            --dataset={dataset} \\
            --predict_file={predict_file} \\
            --dataset_preprocess_dir={dataset_preprocess_dir} \\
            --endpoint={endpoint} {cold_start} {warm_start} \\
        """
    ctx.run(script, echo=True)


@task
def evaluate_simplequestions(
    ctx,
    predict_file,
    dataset_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/simplequestions",
    endpoint="https://query.wikidata.org/sparql",
    cold_start=False,
    warm_start=False,
):
    evaluate_sparkle(
        ctx,
        "simplequestions",
        predict_file,
        dataset_preprocess_dir,
        endpoint,
        cold_start,
        warm_start,
    )


@task
def evaluate_lcquad1(
    ctx,
    predict_file,
    dataset_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/lcquad1",
    endpoint="http://endpoint/sparql",
    cold_start=False,
    warm_start=False,
):
    evaluate_sparkle(
        ctx,
        "lcquad1",
        predict_file,
        dataset_preprocess_dir,
        endpoint,
        cold_start,
        warm_start,
    )


@task
def evaluate_webqsp(
    ctx,
    predict_file,
    dataset_preprocess_dir=f"{os.getenv('DATA_PATH')}/preprocessed/webqsp",
    endpoint="http://endpoint/sparql",
    cold_start=False,
    warm_start=False,
):
    evaluate_sparkle(
        ctx,
        "webqsp",
        predict_file,
        dataset_preprocess_dir,
        endpoint,
        cold_start,
        warm_start,
    )
