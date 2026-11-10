    // SPMTagClassifier: whole-map burden, independent of display smoothing.
    function renderOverallTags(tagRes) {
        renderOverallTagsClassic(tagRes);
        const overall = tagRes && tagRes.overall;
        if (!overall || !overall.percentages) return;
        const tags = overall.tags || [];
        elTagsRow.querySelectorAll('.ov-share').forEach(n => n.remove());
        const meaning = overall.perception.difficultyAvailable
            ? '分支贴合度 × SPM 难度 × 动作物量的积分占比'
            : '分支贴合度 × 动作物量的积分占比（未接入难度）';
        elTagsRow.title = meaning + '；不是正确率或时间占比；各键型可重叠。';
        const label = document.createElement('span');
        label.textContent = '负担占比';
        label.style.cssText = 'font-size:11.5px;opacity:.75;align-self:center';
        elTagsRow.prepend(label);
        if (!tags.length) {
            const unknown = document.createElement('span');
            unknown.textContent = '主要键型未定'; unknown.style.fontSize = '11.5px';
            unknown.title = '现有分支尚未达到整谱门控；不强行补一个标签';
            elTagsRow.appendChild(unknown);
        }
        [...elTagsRow.querySelectorAll('.ov-tag')].forEach((chip, i) => {
            const tag = tags[i], pct = document.createElement('span');
            pct.className = 'pct'; pct.style.marginLeft = '5px';
            if (tag === 'hybrid') {
                pct.textContent = 'RC ' + Math.round(100 * overall.familyMass.rc) + '% / LN '
                    + Math.round(100 * overall.familyMass.ln) + '%';
                chip.title = 'Hybrid：RC / LN 两大家族的负担份额';
            } else {
                pct.textContent = Math.round(overall.labelPercentages[tag] || 0) + '%';
                chip.title = tag === 'rc mix' || tag === 'ln mix'
                    ? '混合标签覆盖的家族负担份额' : meaning;
            }
            chip.appendChild(pct);
        });
        let shown = tags.length;
        for (const [tag, share] of Object.entries(overall.percentages).sort((a,b) => b[1]-a[1])) {
            if (shown >= 4) break;
            if (tags.includes(tag) || share < 1) continue;
            const c = tagColor(tag), chip = document.createElement('span');
            chip.className = 'ov-share'; chip.style.borderColor = hexA(c, CHIP_BORDER_ALPHA);
            chip.style.color = chipInk(c); chip.textContent = tagLabel(tag);
            chip.title = '次要分支的负担份额；未达到整谱 tag 门控';
            const pct = document.createElement('span'); pct.className = 'pct';
            pct.textContent = Math.round(share) + '%'; chip.appendChild(pct);
            elTagsRow.appendChild(chip); shown++;
        }
    }
