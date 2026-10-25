/**
 * Pure client-side Stack v2 + MorphKey runtime for Chelonid-ID (GitHub Pages).
 *
 * Runs 100% in the browser at $0 cost using:
 *   1. Distilled ONNX student (facebook/dinov2-with-registers-small + 384->1024->2048 head)
 *      served from Hugging Face CDN (ajinkyapatil60/turtleid-student-onnx) or ./models/
 *      and cached in Cache Storage for offline field use.
 *   2. Compact float16 browser scoring bundle (./data/stack_v2_browser_bundle.json)
 *      built by core/stack_v2/distill.py:build_browser_bundle.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.TurtleIDBrowser = factory();
    }
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    var DEFAULT_HF_REPO = 'ajinkyapatil60/turtleid-student-onnx';
    var DEFAULT_ONNX_URLS = [
        'https://huggingface.co/' + DEFAULT_HF_REPO + '/resolve/main/turtleid_student_int8.onnx',
        './models/turtleid_student_int8.onnx'
    ];
    var DEFAULT_BUNDLE_URLS = [
        './data/stack_v2_browser_bundle.json',
        'https://huggingface.co/' + DEFAULT_HF_REPO + '/resolve/main/stack_v2_browser_bundle.json'
    ];
    var MODEL_CACHE_NAME = 'turtleid-onnx-v1';

    var IMAGENET_MEAN = [0.485, 0.456, 0.406];
    var IMAGENET_STD = [0.229, 0.224, 0.225];

    var IUCN_LABELS = {
        CR: 'Critically Endangered',
        EN: 'Endangered',
        VU: 'Vulnerable',
        NT: 'Near Threatened',
        LC: 'Least Concern',
        DD: 'Data Deficient',
        EW: 'Extinct in the Wild',
        EX: 'Extinct',
        NE: 'Not Evaluated'
    };

    var TIER_ACTIONS = {
        CONFIRMED: 'Record the species. Retain the photographs as a voucher and attach them to the field register entry.',
        PROBABLE: 'Record as a probable determination. Verify the discriminating character before it goes on any official form.',
        TENTATIVE: 'Insufficient for a species-level record. Report to genus, work through the checklist, and retake photographs if a character is hidden.',
        INDETERMINATE: 'No candidate reaches the reporting threshold. Refer to a chelonian specialist. Do not enter a species on any form.',
        REJECTED: 'The photograph does not resemble the reference set. Retake it (dorsal, ventral, lateral, head) or use the morphological key.'
    };

    var MATRIX_KEYS = [
        'bioclip_mean',
        'dino_mean',
        'cache_mat',
        'local_dino',
        'text_matrix',
        'fam_vecs',
        'gen_vecs',
        'plate_c',
        'biocap_text_matrix',
        'probe_coef',
        'probe_intercept'
    ];

    function base64ToUint8(b64) {
        if (typeof Buffer !== 'undefined') {
            var buf = Buffer.from(b64, 'base64');
            return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
        }
        var binary = atob(b64);
        var bytes = new Uint8Array(binary.length);
        for (var i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    }

    function float16ToFloat32(bits) {
        var sign = (bits & 0x8000) ? -1 : 1;
        var exp = (bits >> 10) & 0x1f;
        var frac = bits & 0x03ff;
        if (exp === 0) {
            if (frac === 0) return sign * 0;
            return sign * Math.pow(2, -14) * (frac / 1024);
        }
        if (exp === 0x1f) {
            return frac ? NaN : sign * Infinity;
        }
        return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }

    function unpackF16(entry) {
        if (!entry) {
            return { data: new Float32Array(0), shape: [0] };
        }
        var b64 = typeof entry.b64 === 'string' ? entry.b64 : (typeof entry.data === 'string' ? entry.data : null);
        if (b64 === null) {
            return { data: new Float32Array(0), shape: [0] };
        }
        var bytes = base64ToUint8(b64);
        var count = bytes.byteLength >> 1;
        var out = new Float32Array(count);
        for (var i = 0; i < count; i++) {
            var lo = bytes[2 * i];
            var hi = bytes[2 * i + 1];
            out[i] = float16ToFloat32(lo | (hi << 8));
        }
        return {
            data: out,
            shape: Array.isArray(entry.shape) ? entry.shape.slice() : [count]
        };
    }

    function normalizeMorphkey(mk) {
        if (!mk) return { characters: {}, matrix: {} };
        var charsObj = {};
        if (Array.isArray(mk.characters)) {
            mk.characters.forEach(function (item) {
                var values = {};
                (item.states || []).forEach(function (st) {
                    values[st.value] = st.label;
                });
                charsObj[item.id] = {
                    label: item.label || item.id,
                    values: values
                };
            });
        } else if (typeof mk.characters === 'object') {
            charsObj = mk.characters;
        }
        var normMatrix = {};
        var srcMat = mk.matrix || {};
        Object.keys(srcMat).forEach(function (sid) {
            var row = srcMat[sid] || {};
            var normRow = {};
            Object.keys(row).forEach(function (ch) {
                var v = row[ch];
                if (v == null) {
                    normRow[ch] = [];
                } else if (Array.isArray(v)) {
                    normRow[ch] = v.slice();
                } else {
                    normRow[ch] = [String(v)];
                }
            });
            normMatrix[sid] = normRow;
        });
        return {
            characters: charsObj,
            characterOrder: Array.isArray(mk.characters)
                ? mk.characters.map(function (c) { return c.id; })
                : Object.keys(charsObj),
            matrix: normMatrix
        };
    }

    function normalizeSpeciesRecord(rec) {
        if (!rec) return {};
        var iucnStatus = (rec.iucn && typeof rec.iucn === 'object')
            ? (rec.iucn.status || '')
            : (rec.iucn_status || rec.iucn || '');
        var iucnLabel = rec.iucn_label || IUCN_LABELS[iucnStatus] || iucnStatus;
        var isThreatened = rec.is_threatened != null
            ? Boolean(rec.is_threatened)
            : (iucnStatus === 'CR' || iucnStatus === 'EN' || iucnStatus === 'VU');
        return {
            id: rec.id || rec.species_id || '',
            common_name: rec.common_en || rec.common_name || rec.id || '',
            common_hindi: rec.common_hi || rec.common_hindi || '',
            scientific_name: rec.scientific_name || '',
            family: rec.family || '',
            iucn_status: iucnStatus,
            iucn_label: iucnLabel,
            is_threatened: isThreatened,
            wpa_schedule: rec.wpa_2022 || rec.wpa_schedule || '',
            cites_appendix: rec.cites || rec.cites_appendix || '',
            states: rec.distribution_states || (rec.distribution && rec.distribution.states_confirmed) || [],
            mp_status: rec.mp_status || 'unknown',
            key_character: rec.key_character || '',
            diagnostics: rec.diagnostics || [],
            confusion_with: rec.confusion_with || [],
            reference_image_url: rec.reference_image || rec.reference_image_url || null
        };
    }

    function unpackBundle(raw) {
        if (!raw || raw._unpacked) return raw;
        var matrices = {};
        var srcMatrices = raw.matrices || raw;
        MATRIX_KEYS.forEach(function (key) {
            if (srcMatrices[key]) {
                matrices[key] = unpackF16(srcMatrices[key]);
            }
        });
        var rawSpecies = raw.species || raw.species_db || {};
        var speciesDb = {};
        Object.keys(rawSpecies).forEach(function (sid) {
            speciesDb[sid] = normalizeSpeciesRecord(rawSpecies[sid]);
        });
        var weights = raw.weights || {};
        return {
            _unpacked: true,
            bundle_version: raw.bundle_version || 1,
            weights: weights,
            thresholds: {
                temperature: weights.temperature != null ? weights.temperature : 0.15,
                gallery_similarity_floor: weights.similarity_floor != null ? weights.similarity_floor : 0.30,
                max_normalised_entropy: weights.max_normalised_entropy != null ? weights.max_normalised_entropy : 0.72,
                tier_confirmed: weights.tier_confirmed != null ? weights.tier_confirmed : 0.85,
                tier_probable: weights.tier_probable != null ? weights.tier_probable : 0.60,
                tier_tentative: weights.tier_tentative != null ? weights.tier_tentative : 0.35
            },
            geo_prior: raw.geo_prior || raw.geo_priors || {
                resident: 1.0,
                introduced: 0.9,
                marginal: 0.35,
                absent: 0.06,
                unknown: 0.6
            },
            confusion_pairs: raw.confusion_pairs || [],
            species_ids: raw.species_ids || [],
            cache_labels: raw.cache_labels || [],
            local_labels: raw.local_labels || [],
            fam_names: raw.fam_names || [],
            gen_names: raw.gen_names || [],
            species_families: raw.species_families || [],
            species_genera: raw.species_genera || [],
            plate_ids: raw.plate_ids || [],
            probe_classes: raw.probe_classes || [],
            matrices: matrices,
            species_db: speciesDb,
            morphkey: normalizeMorphkey(raw.morphkey)
        };
    }

    function l2Normalise(vec) {
        var n = vec.length;
        var sumSq = 0;
        for (var i = 0; i < n; i++) {
            sumSq += vec[i] * vec[i];
        }
        var norm = Math.max(Math.sqrt(sumSq), 1e-12);
        var out = new Float32Array(n);
        for (var j = 0; j < n; j++) {
            out[j] = vec[j] / norm;
        }
        return out;
    }

    function centerQuery(vec, meanTensor) {
        var mean = meanTensor && meanTensor.data ? meanTensor.data : meanTensor;
        var unit = l2Normalise(vec);
        if (!mean || !mean.length) return unit;
        var centered = new Float32Array(unit.length);
        for (var i = 0; i < unit.length; i++) {
            centered[i] = unit[i] - (mean[i] || 0);
        }
        return l2Normalise(centered);
    }

    function dotRow(matrixData, rowOffset, vec, dim) {
        var acc = 0;
        for (var j = 0; j < dim; j++) {
            acc += matrixData[rowOffset + j] * vec[j];
        }
        return acc;
    }

    function tipAdapterScores(query, cacheTensor, cacheLabels, speciesIds, alpha, beta) {
        var nSpecies = speciesIds.length;
        var out = new Float64Array(nSpecies);
        if (!cacheTensor || !cacheTensor.data || !cacheTensor.shape || !cacheTensor.shape[0]) {
            return out;
        }
        var rows = cacheTensor.shape[0];
        var cols = cacheTensor.shape[1] || query.length;
        var data = cacheTensor.data;
        var index = {};
        for (var s = 0; s < nSpecies; s++) {
            index[speciesIds[s]] = s;
        }
        var a = alpha != null ? alpha : 1.0;
        var b = beta != null ? beta : 5.0;
        for (var r = 0; r < rows; r++) {
            var sim = dotRow(data, r * cols, query, cols);
            var affinity = Math.exp(-b * (1.0 - sim));
            var pos = index[cacheLabels[r]];
            if (pos !== undefined) {
                out[pos] += affinity;
            }
        }
        for (var k = 0; k < nSpecies; k++) {
            out[k] *= a;
        }
        return out;
    }

    function softHierarchyBonus(query, famTensor, famNames, genTensor, genNames, speciesFamilies, speciesGenera, wFamily, wGenus) {
        var nSpecies = speciesFamilies.length;
        var bonus = new Float64Array(nSpecies);
        var wf = wFamily != null ? wFamily : 0.15;
        var wg = wGenus != null ? wGenus : 0.20;
        var famScore = {};
        if (famTensor && famTensor.data && famTensor.shape && famTensor.shape[0]) {
            var fCols = famTensor.shape[1] || query.length;
            for (var i = 0; i < famNames.length; i++) {
                famScore[famNames[i]] = dotRow(famTensor.data, i * fCols, query, fCols);
            }
        }
        var genScore = {};
        if (genTensor && genTensor.data && genTensor.shape && genTensor.shape[0]) {
            var gCols = genTensor.shape[1] || query.length;
            for (var j = 0; j < genNames.length; j++) {
                genScore[genNames[j]] = dotRow(genTensor.data, j * gCols, query, gCols);
            }
        }
        for (var s = 0; s < nSpecies; s++) {
            var f = famScore[speciesFamilies[s]] || 0;
            var g = genScore[speciesGenera[s]] || 0;
            bonus[s] = wf * f + wg * g;
        }
        return bonus;
    }

    function plateScores(query, plateTensor, plateIds, speciesIds) {
        var nSpecies = speciesIds.length;
        var scores = new Float64Array(nSpecies);
        for (var i = 0; i < nSpecies; i++) scores[i] = NaN;
        if (!plateTensor || !plateTensor.data || !plateTensor.shape || !plateTensor.shape[0]) {
            return scores;
        }
        var rows = plateTensor.shape[0];
        var cols = plateTensor.shape[1] || query.length;
        var index = {};
        for (var s = 0; s < nSpecies; s++) index[speciesIds[s]] = s;
        for (var r = 0; r < rows; r++) {
            var pos = index[plateIds[r]];
            if (pos === undefined) continue;
            var sim = dotRow(plateTensor.data, r * cols, query, cols);
            if (Number.isNaN(scores[pos]) || sim > scores[pos]) {
                scores[pos] = sim;
            }
        }
        return scores;
    }

    function softmax(logits, temperature) {
        var temp = (temperature != null && temperature > 0) ? temperature : 1.0;
        var n = logits.length;
        var out = new Float64Array(n);
        if (!n) return out;
        var maxVal = -Infinity;
        for (var i = 0; i < n; i++) {
            var scaled = logits[i] / temp;
            if (scaled > maxVal) maxVal = scaled;
        }
        var sum = 0;
        for (var j = 0; j < n; j++) {
            var e = Math.exp(logits[j] / temp - maxVal);
            out[j] = e;
            sum += e;
        }
        if (sum <= 0) return out;
        for (var k = 0; k < n; k++) {
            out[k] /= sum;
        }
        return out;
    }

    function probeProbabilities(fused, coefTensor, interceptTensor, probeClasses, speciesIds) {
        var nSpecies = speciesIds.length;
        var out = new Float64Array(nSpecies);
        if (!coefTensor || !coefTensor.data || !coefTensor.shape || !coefTensor.shape[0] || !probeClasses.length) {
            return out;
        }
        var rows = coefTensor.shape[0];
        var cols = coefTensor.shape[1] || fused.length;
        var intercept = (interceptTensor && interceptTensor.data) ? interceptTensor.data : new Float32Array(rows);
        var logits = new Float64Array(rows);
        for (var r = 0; r < rows; r++) {
            logits[r] = dotRow(coefTensor.data, r * cols, fused, cols) + (intercept[r] || 0);
        }
        var probs = softmax(logits, 1.0);
        var index = {};
        for (var s = 0; s < nSpecies; s++) index[speciesIds[s]] = s;
        for (var c = 0; c < probeClasses.length; c++) {
            var pos = index[probeClasses[c]];
            if (pos !== undefined) {
                out[pos] = probs[c];
            }
        }
        return out;
    }

    function zscoreRow(row) {
        var n = row.length;
        var out = new Float64Array(n);
        var count = 0;
        var sum = 0;
        for (var i = 0; i < n; i++) {
            if (Number.isFinite(row[i])) {
                sum += row[i];
                count++;
            }
        }
        if (count < 2) return out;
        var mean = sum / count;
        var sq = 0;
        for (var j = 0; j < n; j++) {
            if (Number.isFinite(row[j])) {
                var d = row[j] - mean;
                sq += d * d;
            }
        }
        var std = Math.sqrt(sq / count);
        if (std < 1e-6) return out;
        for (var k = 0; k < n; k++) {
            if (Number.isFinite(row[k])) {
                out[k] = (row[k] - mean) / std;
            }
        }
        return out;
    }

    function combineChannels(channels) {
        if (!channels.length) return new Float64Array(0);
        var n = channels[0][0].length;
        var total = new Float64Array(n);
        for (var c = 0; c < channels.length; c++) {
            var row = channels[c][0];
            var weight = channels[c][1];
            if (!weight) continue;
            var z = zscoreRow(row);
            for (var i = 0; i < n; i++) {
                total[i] += weight * z[i];
            }
        }
        return total;
    }

    function maxVisualSimilarity(query, matrices) {
        var best = null;
        for (var m = 0; m < matrices.length; m++) {
            var tensor = matrices[m];
            if (!tensor || !tensor.data || !tensor.shape || !tensor.shape[0]) continue;
            var rows = tensor.shape[0];
            var cols = tensor.shape[1] || query.length;
            for (var r = 0; r < rows; r++) {
                var sim = dotRow(tensor.data, r * cols, query, cols);
                if (best === null || sim > best) best = sim;
            }
        }
        return best;
    }

    function isConfusionPair(a, b, pairs) {
        if (!Array.isArray(pairs)) return false;
        for (var i = 0; i < pairs.length; i++) {
            var p = pairs[i];
            if (Array.isArray(p) && p.length === 2) {
                if ((p[0] === a && p[1] === b) || (p[0] === b && p[1] === a)) {
                    return true;
                }
            }
        }
        return false;
    }

    function fuseChannels(bundle, bioclipQ, dinoQ) {
        var b = unpackBundle(bundle);
        var m = b.matrices;
        var w = b.weights || {};
        var speciesIds = b.species_ids;
        var nSpecies = speciesIds.length;

        var tip = tipAdapterScores(bioclipQ, m.cache_mat, b.cache_labels, speciesIds, 1.15, 5.0);
        var hasDino = m.local_dino && m.local_dino.shape && m.local_dino.shape[0] > 0;
        var tipDino = hasDino
            ? tipAdapterScores(dinoQ, m.local_dino, b.local_labels, speciesIds, 1.0, 5.0)
            : new Float64Array(nSpecies);
        var dinoWeight = hasDino ? (w.tip_dino != null ? w.tip_dino : 0.9) : 0.0;

        var text = new Float64Array(nSpecies);
        if (m.text_matrix && m.text_matrix.data) {
            var tCols = m.text_matrix.shape[1] || bioclipQ.length;
            for (var s = 0; s < nSpecies; s++) {
                text[s] = dotRow(m.text_matrix.data, s * tCols, bioclipQ, tCols);
            }
        }

        var hier = softHierarchyBonus(
            bioclipQ,
            m.fam_vecs,
            b.fam_names,
            m.gen_vecs,
            b.gen_names,
            b.species_families,
            b.species_genera,
            0.15,
            0.20
        );

        var plate = plateScores(bioclipQ, m.plate_c, b.plate_ids, speciesIds);
        var finiteSum = 0;
        var finiteCount = 0;
        for (var p = 0; p < nSpecies; p++) {
            if (Number.isFinite(plate[p])) {
                finiteSum += plate[p];
                finiteCount++;
            }
        }
        var fillVal = finiteCount > 0 ? finiteSum / finiteCount : 0.0;
        var plateRow = new Float64Array(nSpecies);
        for (var q = 0; q < nSpecies; q++) {
            plateRow[q] = Number.isFinite(plate[q]) ? plate[q] : fillVal;
        }

        var fused = new Float32Array(bioclipQ.length + dinoQ.length);
        fused.set(bioclipQ, 0);
        fused.set(dinoQ, bioclipQ.length);
        var logProbs = probeProbabilities(
            fused,
            m.probe_coef,
            m.probe_intercept,
            b.probe_classes,
            speciesIds
        );

        return combineChannels([
            [tip, w.tip != null ? w.tip : 1.25],
            [tipDino, dinoWeight],
            [text, w.text != null ? w.text : 0.55],
            [hier, w.hier != null ? w.hier : 0.35],
            [plateRow, w.plate != null ? w.plate : 0.35],
            [logProbs, w.log_probs != null ? w.log_probs : (w.probe != null ? w.probe : 1.05)]
        ]);
    }

    function applyHeadSpecialist(scores, headBioclipQ, bundle) {
        var b = unpackBundle(bundle);
        var speciesIds = b.species_ids;
        var order = argsortDesc(scores);
        if (order.length < 2 || !headBioclipQ) {
            return { scores: scores, specialist: false };
        }
        var margin = scores[order[0]] - scores[order[1]];
        var marginThreshold = (b.weights && b.weights.margin_specialist != null)
            ? b.weights.margin_specialist
            : 0.15;
        var top1 = speciesIds[order[0]];
        var top2 = speciesIds[order[1]];
        if (margin >= marginThreshold || !isConfusionPair(top1, top2, b.confusion_pairs)) {
            return { scores: scores, specialist: false };
        }
        var headTip = tipAdapterScores(
            headBioclipQ, b.matrices.cache_mat, b.cache_labels, speciesIds, 1.4, 6.0
        );
        var headText = new Float64Array(speciesIds.length);
        var tCols = b.matrices.text_matrix.shape[1] || headBioclipQ.length;
        for (var i = 0; i < speciesIds.length; i++) {
            headText[i] = dotRow(b.matrices.text_matrix.data, i * tCols, headBioclipQ, tCols);
        }
        var boost = combineChannels([
            [headTip, 1.2],
            [headText, 0.8],
            [scores, 0.5]
        ]);
        var allowed = {};
        for (var k = 0; k < Math.min(5, order.length); k++) {
            allowed[speciesIds[order[k]]] = true;
        }
        for (var s = 0; s < speciesIds.length; s++) {
            if (!allowed[speciesIds[s]]) {
                boost[s] = -1e9;
            }
        }
        return { scores: boost, specialist: true };
    }

    function argsortDesc(arr) {
        var idx = [];
        for (var i = 0; i < arr.length; i++) idx.push(i);
        idx.sort(function (a, b) { return arr[b] - arr[a]; });
        return idx;
    }

    function occursInState(speciesRecord, state) {
        if (!speciesRecord || !state) return 'unknown';
        if (state === 'Madhya Pradesh' && speciesRecord.mp_status) {
            return speciesRecord.mp_status;
        }
        var states = speciesRecord.states || [];
        if (states.indexOf(state) !== -1) {
            return speciesRecord.mp_status === 'introduced' ? 'introduced' : 'resident';
        }
        return 'absent';
    }

    function applyGeoPrior(probs, bundle, state) {
        var b = unpackBundle(bundle);
        var n = probs.length;
        var posterior = new Float64Array(n);
        if (!state) {
            for (var i = 0; i < n; i++) posterior[i] = probs[i];
            return posterior;
        }
        var priors = b.geo_prior || { resident: 1.0, introduced: 0.9, marginal: 0.35, absent: 0.06, unknown: 0.6 };
        var sum = 0;
        for (var s = 0; s < n; s++) {
            var sid = b.species_ids[s];
            var rec = b.species_db[sid];
            var occ = occursInState(rec, state);
            var mult = priors[occ] != null ? priors[occ] : (priors.unknown || 0.6);
            posterior[s] = probs[s] * mult;
            sum += posterior[s];
        }
        if (sum <= 0) {
            for (var k = 0; k < n; k++) posterior[k] = probs[k];
            return posterior;
        }
        for (var j = 0; j < n; j++) posterior[j] /= sum;
        return posterior;
    }

    function normalisedEntropy(probs) {
        var n = probs.length;
        if (n <= 1) return 0;
        var h = 0;
        for (var i = 0; i < n; i++) {
            var p = probs[i];
            if (p > 1e-12) {
                h -= p * Math.log(p);
            }
        }
        return h / Math.log(n);
    }

    function assignTier(topPosterior, entropy, rejected, thresholds) {
        if (rejected) return 'REJECTED';
        var t = thresholds || {};
        var confirmed = t.tier_confirmed != null ? t.tier_confirmed : 0.85;
        var probable = t.tier_probable != null ? t.tier_probable : 0.60;
        var tentative = t.tier_tentative != null ? t.tier_tentative : 0.35;
        var maxEntropy = t.max_normalised_entropy != null ? t.max_normalised_entropy : 0.72;
        if (entropy > maxEntropy && topPosterior < confirmed) {
            return 'TENTATIVE';
        }
        if (topPosterior >= confirmed) return 'CONFIRMED';
        if (topPosterior >= probable) return 'PROBABLE';
        if (topPosterior >= tentative) return 'TENTATIVE';
        return 'INDETERMINATE';
    }

    function pairwiseDiscriminator(speciesDb, sidA, sidB) {
        var recA = speciesDb[sidA];
        var recB = speciesDb[sidB];
        if (recA && Array.isArray(recA.confusion_with)) {
            for (var i = 0; i < recA.confusion_with.length; i++) {
                if (recA.confusion_with[i].species_id === sidB) {
                    return recA.confusion_with[i].discriminator || null;
                }
            }
        }
        if (recB && Array.isArray(recB.confusion_with)) {
            for (var j = 0; j < recB.confusion_with.length; j++) {
                if (recB.confusion_with[j].species_id === sidA) {
                    return recB.confusion_with[j].discriminator || null;
                }
            }
        }
        return null;
    }

    /**
     * Port of core/morphkey.py:most_discriminating + verification_questions.
     */
    function mostDiscriminating(morphkey, observations, candidates, limit) {
        var norm = (morphkey && morphkey.characterOrder) ? morphkey : normalizeMorphkey(morphkey);
        var chars = norm.characters || {};
        var orderList = norm.characterOrder || Object.keys(chars);
        var matrix = norm.matrix || {};
        var obs = observations || {};
        var maxQuestions = limit != null ? limit : 3;

        var pool = Array.isArray(candidates) && candidates.length
            ? candidates.slice()
            : Object.keys(matrix);

        var filtered = pool.filter(function (sid) {
            var traits = matrix[sid] || {};
            return Object.keys(obs).every(function (ch) {
                var val = obs[ch];
                if (!val) return true;
                var allowed = traits[ch];
                if (!allowed || !allowed.length) return true;
                return allowed.indexOf(val) !== -1;
            });
        });
        if (filtered.length > 0) {
            pool = filtered;
        }

        var ranked = [];
        orderList.forEach(function (ch) {
            if (obs[ch]) return;
            var meta = chars[ch];
            if (!meta) return;
            var valuesDict = meta.values || {};
            var counts = {};
            var optionSpecies = {};
            var uninformative = [];

            pool.forEach(function (sid) {
                var traits = (matrix[sid] && matrix[sid][ch]) || [];
                if (!traits.length) {
                    uninformative.push(sid);
                    return;
                }
                traits.forEach(function (v) {
                    counts[v] = (counts[v] || 0) + 1;
                    if (!optionSpecies[v]) optionSpecies[v] = [];
                    optionSpecies[v].push(sid);
                });
            });

            var activeValues = Object.keys(counts);
            if (activeValues.length < 2) return;

            var total = 0;
            activeValues.forEach(function (v) { total += counts[v]; });
            if (total <= 0) return;

            var entropy = 0;
            activeValues.forEach(function (v) {
                var p = counts[v] / total;
                if (p > 0) entropy -= p * Math.log2(p);
            });

            var options = [];
            Object.keys(valuesDict).forEach(function (valKey) {
                var matching = optionSpecies[valKey] || [];
                if (!matching.length) return;
                options.push({
                    value: valKey,
                    label: valuesDict[valKey],
                    species_ids: matching,
                    uninformative_species_ids: uninformative
                });
            });

            if (options.length >= 2) {
                ranked.push({
                    character: ch,
                    label: meta.label || ch,
                    prompt: meta.label || ch,
                    entropy: entropy,
                    options: options
                });
            }
        });

        ranked.sort(function (a, b) { return b.entropy - a.entropy; });
        return ranked.slice(0, maxQuestions);
    }

    function scoreEmbeddings(bundle, rawBioclip, rawDino, options) {
        var opts = options || {};
        var b = unpackBundle(bundle);
        var state = opts.state || '';
        var cropMethod = opts.crop_method || 'border_crop';
        var engineName = opts.engine || 'stack_v2_browser_onnx';

        var bioclipQ = opts.alreadyCentered
            ? l2Normalise(rawBioclip)
            : centerQuery(rawBioclip, b.matrices.bioclip_mean);
        var dinoQ = opts.alreadyCentered
            ? l2Normalise(rawDino)
            : centerQuery(rawDino, b.matrices.dino_mean);

        var scores = fuseChannels(b, bioclipQ, dinoQ);
        var specialist = false;
        if (opts.rawHeadBioclip) {
            var headQ = opts.alreadyCentered
                ? l2Normalise(opts.rawHeadBioclip)
                : centerQuery(opts.rawHeadBioclip, b.matrices.bioclip_mean);
            var spec = applyHeadSpecialist(scores, headQ, b);
            scores = spec.scores;
            specialist = spec.specialist;
        }

        var maxSim = maxVisualSimilarity(bioclipQ, [b.matrices.cache_mat, b.matrices.plate_c]);
        var thresholds = b.thresholds || {};
        var temp = thresholds.temperature != null ? thresholds.temperature : 0.15;
        var probs = softmax(scores, temp);
        var posterior = applyGeoPrior(probs, b, state);

        var order = argsortDesc(posterior);
        var scoreOrder = argsortDesc(scores);
        var margin = scoreOrder.length >= 2 ? (scores[scoreOrder[0]] - scores[scoreOrder[1]]) : 0.0;

        var floor = thresholds.gallery_similarity_floor != null ? thresholds.gallery_similarity_floor : 0.30;
        var rejected = (maxSim !== null && maxSim < floor && opts.enableSimGate !== false);
        var entropy = normalisedEntropy(posterior);
        var topPosterior = order.length ? posterior[order[0]] : 0.0;
        var tier = assignTier(topPosterior, entropy, rejected, thresholds);

        var warnings = [];
        warnings.push(
            'Model probabilities are UNCALIBRATED (softmax of fused Stack v2 scores). ' +
            'Treat them as a ranking aid and confirm every determination against the morphological checklist.'
        );

        var topIds = order.slice(0, 3).map(function (idx) { return b.species_ids[idx]; });
        var pairwise = null;
        var marginSpecialist = (b.weights && b.weights.margin_specialist != null) ? b.weights.margin_specialist : 0.15;

        if (topIds.length >= 2) {
            pairwise = pairwiseDiscriminator(b.species_db, topIds[0], topIds[1]);
            var topRec = b.species_db[topIds[0]] || {};
            var runnerRec = b.species_db[topIds[1]] || {};
            var runnerP = posterior[order[1]];
            if (pairwise && (runnerP > 0.15 || margin < marginSpecialist)) {
                warnings.push('Close call against ' + (runnerRec.scientific_name || topIds[1]) + '. Check this character: ' + pairwise);
            }
            if (isConfusionPair(topIds[0], topIds[1], b.confusion_pairs) && margin < marginSpecialist) {
                warnings.push(
                    'Low-margin confusion pair: ' + (topRec.common_name || topIds[0]) +
                    ' vs ' + (runnerRec.common_name || topIds[1]) + '. Confirm the checklist before recording a species.'
                );
            }
            if (topRec.wpa_schedule && runnerRec.wpa_schedule && topRec.wpa_schedule !== runnerRec.wpa_schedule && (runnerP > 0.15 || margin < marginSpecialist)) {
                warnings.push(
                    'LEGAL DIVERGENCE: ' + topRec.scientific_name + ' is ' + topRec.wpa_schedule +
                    ' but ' + runnerRec.scientific_name + ' is ' + runnerRec.wpa_schedule +
                    '. The offence category differs between the top two candidates. Do not finalise paperwork on this determination alone.'
                );
            }
        }

        if (topIds.length >= 1) {
            var primary = b.species_db[topIds[0]] || {};
            var occ = occursInState(primary, state);
            if (state && occ === 'absent') {
                warnings.push(
                    primary.scientific_name + ' is not known from ' + state +
                    '. If the determination holds, this is a trade, transport or release record and should be reported as such, not logged as a wild occurrence.'
                );
            } else if (state && occ === 'marginal') {
                warnings.push(
                    primary.scientific_name + ' is marginal in ' + state +
                    '. Retain voucher photographs before recording.'
                );
            }
            var probableFloor = thresholds.tier_probable != null ? thresholds.tier_probable : 0.60;
            if (primary.is_threatened && topPosterior >= probableFloor && tier !== 'REJECTED') {
                warnings.push(
                    primary.scientific_name + ' is ' + (primary.iucn_label || primary.iucn_status) +
                    ' (' + primary.iucn_status + ') and ' + primary.wpa_schedule +
                    '. Notify the Divisional Forest Officer and retain the record regardless of outcome.'
                );
            }
        }
        if (specialist) {
            warnings.push('Head-crop specialist re-ranked a known confusion pair (margin was under ' + marginSpecialist.toFixed(2) + ').');
        }

        var questions = topIds.length >= 2 ? mostDiscriminating(b.morphkey, {}, topIds, 3) : [];
        var matches = order.slice(0, 3).map(function (pos) {
            var sid = b.species_ids[pos];
            var sp = b.species_db[sid] || {};
            var conf = Number(posterior[pos]);
            return {
                species_id: sid,
                common_name: sp.common_name || sid,
                common_hindi: sp.common_hindi || null,
                scientific_name: sp.scientific_name || sid,
                confidence: conf,
                score: conf,
                raw_score: Number(scores[pos]),
                family: sp.family || '',
                iucn_status: sp.iucn_status || '',
                iucn_label: sp.iucn_label || '',
                wpa_schedule: sp.wpa_schedule || '',
                cites_appendix: sp.cites_appendix || '',
                key_character: sp.key_character || '',
                diagnostics: sp.diagnostics || [],
                state_occurrence: state ? occursInState(sp, state) : 'unknown',
                reference_image_url: sp.reference_image_url || null
            };
        });

        var abstain = (tier === 'TENTATIVE' || tier === 'INDETERMINATE' || tier === 'REJECTED');
        return {
            status: 'success',
            engine: engineName,
            tier: tier,
            abstain: abstain,
            warnings: warnings,
            pairwise_discriminator: pairwise,
            verification_questions: questions,
            top_match: matches[0] || null,
            top_matches: matches,
            matches: matches,
            message: TIER_ACTIONS[tier] || '',
            margin: margin,
            specialist: specialist,
            crop_method: cropMethod,
            max_similarity: maxSim
        };
    }

    /**
     * Port of core/foreground.py:crop_to_animal to HTML5 Canvas.
     */
    function cropToAnimalCanvas(imageOrCanvas, pad) {
        var padding = pad != null ? pad : 0.12;
        var srcW = imageOrCanvas.naturalWidth || imageOrCanvas.width;
        var srcH = imageOrCanvas.naturalHeight || imageOrCanvas.height;
        var fullCanvas = document.createElement('canvas');
        fullCanvas.width = srcW;
        fullCanvas.height = srcH;
        var ctx = fullCanvas.getContext('2d');
        ctx.drawImage(imageOrCanvas, 0, 0, srcW, srcH);

        if (srcW < 32 || srcH < 32) {
            return { animal: fullCanvas, head: headFromAnimalCanvas(fullCanvas), method: 'border_crop', box: [0, 0, srcW, srcH] };
        }

        var maxSide = 256;
        var scale = Math.min(1, maxSide / Math.max(srcW, srcH));
        var w = Math.max(32, Math.round(srcW * scale));
        var h = Math.max(32, Math.round(srcH * scale));
        var work = document.createElement('canvas');
        work.width = w;
        work.height = h;
        var wctx = work.getContext('2d');
        wctx.drawImage(fullCanvas, 0, 0, w, h);
        var pixels = wctx.getImageData(0, 0, w, h).data;

        var border = Math.max(4, Math.floor(0.05 * Math.min(h, w)));
        var rs = [], gs = [], bs = [];
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                if (y < border || y >= h - border || x < border || x >= w - border) {
                    var p = (y * w + x) * 4;
                    rs.push(pixels[p]);
                    gs.push(pixels[p + 1]);
                    bs.push(pixels[p + 2]);
                }
            }
        }
        function median(arr) {
            var copy = arr.slice().sort(function (a, b) { return a - b; });
            return copy[copy.length >> 1] || 0;
        }
        var bgR = median(rs), bgG = median(gs), bgB = median(bs);
        var borderDists = [];
        for (var i = 0; i < rs.length; i++) {
            var dr = rs[i] - bgR, dg = gs[i] - bgG, db = bs[i] - bgB;
            borderDists.push(Math.sqrt(dr * dr + dg * dg + db * db));
        }
        borderDists.sort(function (a, b) { return a - b; });
        var p90 = borderDists[Math.floor(0.9 * borderDists.length)] || 0;
        var threshold = Math.max(32.0, p90 + 24.0);

        var top = h, bottom = -1, left = w, right = -1, active = 0;
        for (var py = 0; py < h; py++) {
            for (var px = 0; px < w; px++) {
                var idx = (py * w + px) * 4;
                var dR = pixels[idx] - bgR;
                var dG = pixels[idx + 1] - bgG;
                var dB = pixels[idx + 2] - bgB;
                var dist = Math.sqrt(dR * dR + dG * dG + dB * dB);
                if (dist > threshold) {
                    active++;
                    if (py < top) top = py;
                    if (py > bottom) bottom = py;
                    if (px < left) left = px;
                    if (px > right) right = px;
                }
            }
        }
        var fraction = active / (w * h);
        if (fraction < 0.02 || fraction > 0.85 || bottom <= top || right <= left) {
            return { animal: fullCanvas, head: headFromAnimalCanvas(fullCanvas), method: 'border_crop', box: [0, 0, srcW, srcH] };
        }
        var boxH = bottom - top + 1;
        var boxW = right - left + 1;
        if (boxH < 0.12 * h || boxW < 0.12 * w) {
            return { animal: fullCanvas, head: headFromAnimalCanvas(fullCanvas), method: 'border_crop', box: [0, 0, srcW, srcH] };
        }
        var padY = Math.floor(padding * boxH);
        var padX = Math.floor(padding * boxW);
        var cTop = Math.max(0, Math.floor((top - padY) / scale));
        var cLeft = Math.max(0, Math.floor((left - padX) / scale));
        var cBottom = Math.min(srcH, Math.ceil((bottom + padY + 1) / scale));
        var cRight = Math.min(srcW, Math.ceil((right + padX + 1) / scale));
        var cW = Math.max(16, cRight - cLeft);
        var cH = Math.max(16, cBottom - cTop);

        var animalCanvas = document.createElement('canvas');
        animalCanvas.width = cW;
        animalCanvas.height = cH;
        animalCanvas.getContext('2d').drawImage(fullCanvas, cLeft, cTop, cW, cH, 0, 0, cW, cH);
        return {
            animal: animalCanvas,
            head: headFromAnimalCanvas(animalCanvas),
            method: 'border_crop',
            box: [cLeft, cTop, cW, cH]
        };
    }

    function headFromAnimalCanvas(animalCanvas) {
        var w = animalCanvas.width;
        var h = Math.max(32, Math.floor(0.38 * animalCanvas.height));
        var head = document.createElement('canvas');
        head.width = w;
        head.height = h;
        head.getContext('2d').drawImage(animalCanvas, 0, 0, w, h, 0, 0, w, h);
        return head;
    }

    function canvasToTensorNCHW(canvas, flipHorizontal) {
        var size = 224;
        var tmp = document.createElement('canvas');
        tmp.width = size;
        tmp.height = size;
        var ctx = tmp.getContext('2d');
        if (flipHorizontal) {
            ctx.translate(size, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(canvas, 0, 0, size, size);
        var rgba = ctx.getImageData(0, 0, size, size).data;
        var out = new Float32Array(3 * size * size);
        var plane = size * size;
        for (var i = 0; i < plane; i++) {
            var r = rgba[i * 4] / 255.0;
            var g = rgba[i * 4 + 1] / 255.0;
            var b = rgba[i * 4 + 2] / 255.0;
            out[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            out[plane + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            out[2 * plane + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
        return out;
    }

    async function fetchModelBufferWithCache(urls, onStatus) {
        var list = Array.isArray(urls) ? urls : [urls];
        var lastError = null;
        for (var i = 0; i < list.length; i++) {
            var url = list[i];
            if (!url) continue;
            try {
                if (typeof caches !== 'undefined') {
                    var cache = await caches.open(MODEL_CACHE_NAME);
                    var cached = await cache.match(url);
                    if (cached && cached.ok) {
                        if (onStatus) onStatus('Loading cached browser vision model…');
                        return { buffer: await cached.arrayBuffer(), url: url, cached: true };
                    }
                }
                if (onStatus) onStatus('Downloading free browser vision model (~23 MB, cached after first run)…');
                var response = await fetch(url, { mode: 'cors' });
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status + ' for ' + url);
                }
                if (typeof caches !== 'undefined') {
                    try {
                        var cacheStore = await caches.open(MODEL_CACHE_NAME);
                        await cacheStore.put(url, response.clone());
                    } catch (e) {}
                }
                return { buffer: await response.arrayBuffer(), url: url, cached: false };
            } catch (err) {
                lastError = err;
            }
        }
        throw lastError || new Error('Unable to load ONNX student model.');
    }

    var _cachedBundle = null;
    var _cachedSession = null;
    var _cachedModelUrl = null;

    async function loadBrowserBundle(urls) {
        if (_cachedBundle) return _cachedBundle;
        var list = Array.isArray(urls) ? urls : (urls ? [urls] : DEFAULT_BUNDLE_URLS);
        var lastErr = null;
        for (var i = 0; i < list.length; i++) {
            try {
                var resp = await fetch(list[i]);
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                var raw = await resp.json();
                _cachedBundle = unpackBundle(raw);
                return _cachedBundle;
            } catch (err) {
                lastErr = err;
            }
        }
        throw lastErr || new Error('Unable to load Stack v2 browser bundle');
    }

    async function loadOnnxSession(onnxUrls, onStatus) {
        if (_cachedSession) return { session: _cachedSession, url: _cachedModelUrl };
        var ort = (typeof window !== 'undefined' && window.ort) ? window.ort : null;
        if (!ort) {
            throw new Error('onnxruntime-web (window.ort) is not loaded');
        }
        var loaded = await fetchModelBufferWithCache(onnxUrls || DEFAULT_ONNX_URLS, onStatus);
        var providers = ['wasm'];
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            providers = ['webgpu', 'wasm'];
        }
        _cachedSession = await ort.InferenceSession.create(loaded.buffer, {
            executionProviders: providers
        });
        _cachedModelUrl = loaded.url;
        return { session: _cachedSession, url: _cachedModelUrl };
    }

    async function embedCanvasWithStudent(session, canvas) {
        var ort = window.ort;
        var t1 = canvasToTensorNCHW(canvas, false);
        var t2 = canvasToTensorNCHW(canvas, true);
        var tensor1 = new ort.Tensor('float32', t1, [1, 3, 224, 224]);
        var tensor2 = new ort.Tensor('float32', t2, [1, 3, 224, 224]);

        var out1 = await session.run({ pixel_values: tensor1 });
        var out2 = await session.run({ pixel_values: tensor2 });

        var b1 = out1.bioclip_emb.data;
        var b2 = out2.bioclip_emb.data;
        var d1 = out1.dino_emb.data;
        var d2 = out2.dino_emb.data;

        var bAvg = new Float32Array(b1.length);
        for (var i = 0; i < b1.length; i++) bAvg[i] = 0.5 * (b1[i] + b2[i]);
        var dAvg = new Float32Array(d1.length);
        for (var j = 0; j < d1.length; j++) dAvg[j] = 0.5 * (d1[j] + d2[j]);

        return {
            bioclip: l2Normalise(bAvg),
            dino: l2Normalise(dAvg)
        };
    }

    function loadImageElement(src) {
        return new Promise(function (resolve, reject) {
            var img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function () { resolve(img); };
            img.onerror = function () { reject(new Error('Could not decode uploaded photograph')); };
            img.src = src;
        });
    }

    async function identifyInBrowser(options) {
        var opts = options || {};
        var onStatus = opts.onStatus || function () {};
        onStatus('Loading Stack v2 species bundle…');
        var bundle = await loadBrowserBundle(opts.bundleUrls);
        var modelInfo = await loadOnnxSession(opts.onnxUrls, onStatus);

        onStatus('Cropping animal & running Stack v2 vision model in browser…');
        var img = opts.imageElement || await loadImageElement(opts.imageDataUrl);
        var crops = cropToAnimalCanvas(img, 0.12);
        var emb = await embedCanvasWithStudent(modelInfo.session, crops.animal);

        var prelim = scoreEmbeddings(bundle, emb.bioclip, emb.dino, {
            state: opts.state || '',
            crop_method: crops.method,
            alreadyCentered: true,
            engine: 'stack_v2_browser_onnx'
        });

        if (
            prelim.top_matches &&
            prelim.top_matches.length >= 2 &&
            prelim.margin < ((bundle.weights && bundle.weights.margin_specialist) || 0.15) &&
            isConfusionPair(prelim.top_matches[0].species_id, prelim.top_matches[1].species_id, bundle.confusion_pairs)
        ) {
            onStatus('Close lookalike pair detected — running head-crop specialist…');
            var headEmb = await embedCanvasWithStudent(modelInfo.session, crops.head);
            return scoreEmbeddings(bundle, emb.bioclip, emb.dino, {
                state: opts.state || '',
                crop_method: crops.method,
                rawHeadBioclip: headEmb.bioclip,
                alreadyCentered: true,
                engine: 'stack_v2_browser_onnx'
            });
        }
        return prelim;
    }

    return {
        DEFAULT_HF_REPO: DEFAULT_HF_REPO,
        DEFAULT_ONNX_URLS: DEFAULT_ONNX_URLS,
        DEFAULT_BUNDLE_URLS: DEFAULT_BUNDLE_URLS,
        unpackF16: unpackF16,
        unpackBundle: unpackBundle,
        l2Normalise: l2Normalise,
        centerQuery: centerQuery,
        tipAdapterScores: tipAdapterScores,
        softHierarchyBonus: softHierarchyBonus,
        plateScores: plateScores,
        probeProbabilities: probeProbabilities,
        zscoreRow: zscoreRow,
        combineChannels: combineChannels,
        fuseChannels: fuseChannels,
        applyHeadSpecialist: applyHeadSpecialist,
        applyGeoPrior: applyGeoPrior,
        normalisedEntropy: normalisedEntropy,
        assignTier: assignTier,
        mostDiscriminating: mostDiscriminating,
        scoreEmbeddings: scoreEmbeddings,
        cropToAnimalCanvas: cropToAnimalCanvas,
        headFromAnimalCanvas: headFromAnimalCanvas,
        canvasToTensorNCHW: canvasToTensorNCHW,
        loadBrowserBundle: loadBrowserBundle,
        loadOnnxSession: loadOnnxSession,
        identifyInBrowser: identifyInBrowser
    };
});
